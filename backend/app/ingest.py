from __future__ import annotations
import io
import re
import uuid
from typing import List, Tuple, Dict, Any
import pandas as pd
from sqlalchemy import Table, Column, MetaData, String, Float, Integer, DateTime, TEXT
from sqlalchemy.engine import Engine
import os
from .db import store_documents

# Utility helpers ---------------------------------------------------

def _slugify(name: str) -> str:
    """Make a safe identifier (lowercase, underscores, alphanumeric)."""
    name = name.strip() if isinstance(name, str) else ""
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return (name or "col").lower()


def _infer_sqlalchemy_type(series: pd.Series):
    """Infer a SQLAlchemy column type from a pandas series."""
    if pd.api.types.is_integer_dtype(series):
        return Integer
    if pd.api.types.is_float_dtype(series):
        return Float
    if pd.api.types.is_datetime64_any_dtype(series):
        return DateTime
    # Use TEXT for PostgreSQL compatibility (no length limit)
    return TEXT


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic cleaning:
    - detect header row (first row with >50% non-null)
    - rename unnamed columns
    - drop fully empty rows/cols
    - strip strings
    - try to coerce date-like columns
    """
    if df is None:
        return pd.DataFrame()

    # Drop entirely empty rows/cols early
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # If the DataFrame has unnamed columns like 'Unnamed: 0' or numeric column labels,
    # attempt to detect header row in the first few rows.
    header_idx = None
    scan_rows = min(10, len(df))
    for i in range(scan_rows):
        row = df.iloc[i]
        # count non-empty cells in this row
        non_null_ratio = row.notna().mean() if len(df.columns) else 0
        # prefer rows that look string-y and not pure numbers, and with at least half columns non-null
        if non_null_ratio >= 0.5:
            header_idx = i
            break

    if header_idx is not None and header_idx != 0:
        # treat header row as column names
        new_cols = df.iloc[header_idx].fillna("").astype(str).tolist()
        df = df.iloc[header_idx + 1 :].reset_index(drop=True)
        df.columns = new_cols

    # If columns look like 'Unnamed' or integer indices, use first row as header if it's not empty
    if any(str(c).lower().startswith("unnamed") or str(c).isdigit() for c in df.columns):
        if len(df) > 0:
            first = df.iloc[0]
            if first.notna().sum() >= max(1, int(0.5 * len(df.columns))):
                new_cols = first.fillna("").astype(str).tolist()
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = new_cols

    # Fill missing column names, ensure uniqueness
    new_cols = []
    seen = set()
    for i, c in enumerate(df.columns):
        if c is None or str(c).strip() == "":
            base = f"col_{i}"
        else:
            base = str(c)
        name = _slugify(base)
        # ensure uniqueness
        suffix = 1
        while name in seen:
            suffix += 1
            name = f"{_slugify(base)}_{suffix}"
        seen.add(name)
        new_cols.append(name)
    df.columns = new_cols

    # strip string values
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": None, "None": None})

    # attempt to coerce date-like columns
    for c in df.columns:
        if df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                if parsed.notna().sum() and parsed.notna().sum() >= 0.3 * len(parsed):
                    df[c] = parsed
            except Exception:
                pass

    return df


# Main ingest function ---------------------------------------------------

def ingest_excel(engine: Engine, filename: str, content: bytes) -> Tuple[str, List[str]]:
    """
    Ingest an Excel file (bytes) into the provided SQLAlchemy engine.
    Creates a dataset id and a table per sheet named <dataset_id>__<slug_sheet>.
    Returns (dataset_id, sheet_names).
    """
    dataset_id = uuid.uuid4().hex[:8]

    # let pandas choose engine; sometimes need openpyxl for xlsx
    xls = pd.ExcelFile(io.BytesIO(content), engine=None)
    meta = MetaData()
    sheet_names: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    # Check if using PostgreSQL
    is_postgres = engine.url.drivername == 'postgresql'

    with engine.begin() as conn:
        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet)
            except Exception:
                # fallback parsing per-sheet
                try:
                    df = pd.read_excel(io.BytesIO(content), sheet_name=sheet, engine="openpyxl")
                except Exception:
                    df = pd.DataFrame()
            # clean
            df = _clean_dataframe(df)

            # define columns for SQLAlchemy Table
            cols = []
            for name in df.columns:
                col_type = _infer_sqlalchemy_type(df[name]) if name in df.columns else TEXT
                cols.append(Column(name, col_type))
            
            table_name = f"{dataset_id}__{_slugify(sheet) or 'sheet'}"
            
            # For PostgreSQL, use lowercase table names
            if is_postgres:
                table_name = table_name.lower()
            
            table = Table(table_name, meta, *cols)
            
            # drop/create for idempotency
            try:
                table.drop(bind=conn, checkfirst=True)
            except Exception:
                pass
            try:
                table.create(bind=conn, checkfirst=True)
            except Exception:
                pass

            # write rows using pandas to_sql
            if not df.empty:
                try:
                    # For PostgreSQL, we need to handle the connection differently
                    df.to_sql(
                        table_name, 
                        con=conn, 
                        if_exists="append", 
                        index=False,
                        method='multi',  # Faster bulk inserts
                        chunksize=1000  # Insert in chunks
                    )
                except Exception as e:
                    # Fallback: try row by row insertion
                    print(f"Bulk insert failed for {table_name}, trying row by row: {e}")
                    try:
                        for _, row in df.iterrows():
                            try:
                                pd.DataFrame([row]).to_sql(
                                    table_name, 
                                    con=conn, 
                                    if_exists="append", 
                                    index=False
                                )
                            except Exception:
                                continue
                    except Exception:
                        pass

            slug = _slugify(sheet) or "sheet"
            sheet_names.append(slug)

            # prepare small document store for vector DB: columns and some sample rows
            for col in df.columns:
                documents.append(f"column: {col} in table {table_name}")
                metadatas.append({"type": "column", "table": table_name, "column": col})
                ids.append(f"{table_name}__col__{col}")

            for i, (_, row) in enumerate(df.head(20).iterrows()):
                try:
                    row_text = "; ".join([f"{c}={row[c]}" for c in df.columns])
                except Exception:
                    row_text = "; ".join([f"{c}={str(row.get(c,''))}" for c in df.columns])
                documents.append(f"row in {table_name}: {row_text}")
                metadatas.append({"type": "row", "table": table_name, "index": int(i)})
                ids.append(f"{table_name}__row__{i}")

    # store documents (best-effort)
    if documents:
        try:
            store_documents(dataset_id, documents, metadatas, ids)
        except Exception:
            pass

    return dataset_id, sheet_names