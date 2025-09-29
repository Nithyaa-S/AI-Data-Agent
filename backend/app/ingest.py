from __future__ import annotations
import io
import re
import uuid
import logging
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from sqlalchemy import Table, Column, MetaData, String, Float, Integer, DateTime, TEXT, text, DDL, event
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError
import os
from .db import store_documents

logger = logging.getLogger(__name__)

def ensure_postgres_schema(conn: Connection):
    """Ensure the PostgreSQL schema exists and is accessible."""
    try:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS public"))
        conn.execute(text("SET search_path TO public"))
        conn.commit()
    except Exception as e:
        logger.error(f"Error ensuring PostgreSQL schema: {e}")
        raise

def safe_table_name(name: str, is_postgres: bool = False) -> str:
    """Convert a string to a safe table name."""
    # Convert to lowercase for PostgreSQL
    if is_postgres:
        name = name.lower()
    # Replace special characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Ensure it starts with a letter
    if not name or not name[0].isalpha():
        name = 't_' + (name or 'table')
    # Ensure name is not too long (PostgreSQL has a 63-byte limit)
    return name[:63]

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
    if series.empty:
        return TEXT
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
    if df is None or df.empty:
        return pd.DataFrame()

    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Drop entirely empty rows/cols early
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    
    if df.empty:
        return df

    # If the DataFrame has unnamed columns like 'Unnamed: 0' or numeric column labels,
    # attempt to detect header row in the first few rows.
    header_idx = None
    scan_rows = min(10, len(df))
    
    for i in range(scan_rows):
        row = df.iloc[i]
        # count non-empty cells in this row
        non_null_ratio = row.notna().mean() if len(df.columns) > 0 else 0
        # prefer rows that look string-y and not pure numbers, and with at least half columns non-null
        if non_null_ratio >= 0.5:
            header_idx = i
            break

    if header_idx is not None and header_idx != 0:
        # treat header row as column names
        new_cols = df.iloc[header_idx].fillna("").astype(str).tolist()
        df = df.iloc[header_idx + 1:].reset_index(drop=True)
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
        orig_name = name
        while name in seen:
            name = f"{orig_name}_{suffix}"
            suffix += 1
        seen.add(name)
        new_cols.append(name)
    
    df.columns = new_cols

    # strip string values
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": None, "None": None, "": None})

    # attempt to coerce date-like columns
    for c in df.columns:
        if df[c].dtype == object and not df[c].empty:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                if parsed.notna().sum() >= 0.3 * len(parsed):
                    df[c] = parsed
            except Exception:
                pass

    return df

def ingest_excel(engine: Engine, filename: str, content: bytes) -> Tuple[str, List[str]]:
    """
    Ingest an Excel file into the database.
    
    Args:
        engine: SQLAlchemy engine
        filename: Original filename of the uploaded file
        content: File content as bytes
        
    Returns:
        Tuple of (dataset_id, list of sheet names)
    """
    from sqlalchemy import inspect, exc
    
    # Generate a unique dataset ID (lowercase for PostgreSQL compatibility)
    dataset_id = f'ds_{uuid.uuid4().hex[:10]}'.lower()
    
    try:
        # Read the Excel file
        try:
            xls = pd.ExcelFile(io.BytesIO(content), engine='openpyxl')
            sheet_names = xls.sheet_names
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise ValueError(f"Failed to read Excel file: {e}")
        
        if not sheet_names:
            raise ValueError("Excel file contains no sheets")
        
        # Store metadata and process each sheet
        meta = MetaData()
        documents = []
        metadatas = []
        ids = []
        processed_sheet_names = []
        
        # Check database type
        is_postgres = engine.url.drivername == 'postgresql'
        
        # For PostgreSQL, ensure the schema exists and is accessible
        with engine.connect() as conn:
            if is_postgres:
                ensure_postgres_schema(conn)
            
            for sheet in sheet_names:
                try:
                    # Read the sheet
                    try:
                        df = pd.read_excel(io.BytesIO(content), sheet_name=sheet, engine='openpyxl')
                    except Exception as e:
                        logger.warning(f"Failed to read sheet '{sheet}': {e}")
                        continue
                    
                    # Clean the dataframe
                    df = _clean_dataframe(df)
                    
                    if df.empty:
                        logger.warning(f"Sheet '{sheet}' is empty after cleaning, skipping")
                        continue
                    
                    # Define columns for SQLAlchemy Table
                    cols = []
                    for name in df.columns:
                        col_type = _infer_sqlalchemy_type(df[name]) if not df.empty and name in df.columns else TEXT
                        cols.append(Column(name, col_type))
                    
                    # Create a safe table name
                    table_name = f"{dataset_id}__{_slugify(sheet) or 'sheet'}"
                    table_name = safe_table_name(table_name, is_postgres)
                    
                    # Create the table
                    table = Table(table_name, meta, *cols)
                    
                    # Drop existing table if it exists
                    try:
                        table.drop(engine, checkfirst=True)
                    except Exception as e:
                        logger.warning(f"Error dropping table {table_name}: {e}")
                    
                    # Create the table
                    try:
                        table.create(engine)
                    except Exception as e:
                        logger.error(f"Error creating table {table_name}: {e}")
                        raise
                    
                    # Insert data into the table
                    if not df.empty:
                        try:
                            # Convert datetime columns to string for SQLite compatibility
                            for col in df.select_dtypes(include=['datetime64[ns]']).columns:
                                df[col] = df[col].astype(str)
                            
                            # Insert data in chunks
                            df.to_sql(
                                name=table_name,
                                con=engine,
                                if_exists='append',
                                index=False,
                                chunksize=1000,
                                method='multi' if not is_postgres else None  # method='multi' can cause issues with PostgreSQL
                            )
                            logger.info(f"Successfully inserted {len(df)} rows into {table_name}")
                            
                        except Exception as e:
                            logger.error(f"Error inserting data into {table_name}: {e}")
                            # Try row by row as fallback
                            try:
                                df.to_sql(
                                    name=table_name,
                                    con=engine,
                                    if_exists='append',
                                    index=False,
                                    chunksize=1,
                                    method=None
                                )
                                logger.info(f"Successfully inserted {len(df)} rows into {table_name} (row by row)")
                            except Exception as e2:
                                logger.error(f"Row-by-row insert failed for {table_name}: {e2}")
                                raise
                    
                    # Add to processed sheets
                    slug = _slugify(sheet) or "sheet"
                    processed_sheet_names.append(slug)
                    
                    # Prepare documents for vector store
                    if not df.empty:
                        # Add column information
                        for col in df.columns:
                            doc_id = f"{table_name}__col__{_slugify(col)}"
                            doc_text = f"column: {col} in table {table_name}"
                            documents.append(doc_text)
                            metadatas.append({
                                "type": "column",
                                "table": table_name,
                                "column": col,
                                "dataset_id": dataset_id
                            })
                            ids.append(doc_id)
                        
                        # Add sample rows (first 20 rows)
                        sample_rows = df.head(20)
                        for i, (_, row) in enumerate(sample_rows.iterrows()):
                            try:
                                row_values = []
                                for col in df.columns:
                                    val = row[col]
                                    if pd.isna(val):
                                        val = ""
                                    row_values.append(f"{col}={val}")
                                row_text = "; ".join(row_values)
                                
                                doc_id = f"{table_name}__row__{i}"
                                doc_text = f"row in {table_name}: {row_text}"
                                
                                documents.append(doc_text)
                                metadatas.append({
                                    "type": "row",
                                    "table": table_name,
                                    "index": int(i),
                                    "dataset_id": dataset_id
                                })
                                ids.append(doc_id)
                            except Exception as e:
                                logger.warning(f"Error processing row {i} in {table_name}: {e}")
                                continue
                
                except Exception as e:
                    logger.error(f"Error processing sheet '{sheet}': {e}", exc_info=True)
                    continue
            
            # Store documents in vector store
            if documents:
                try:
                    store_documents(dataset_id, documents, metadatas, ids)
                    logger.info(f"Stored {len(documents)} documents in vector store for dataset {dataset_id}")
                except Exception as e:
                    logger.error(f"Error storing documents in vector store: {e}", exc_info=True)
                    raise
        
        return dataset_id, processed_sheet_names
        
    except Exception as e:
        logger.error(f"Error in ingest_excel: {e}", exc_info=True)
        raise