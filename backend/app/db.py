from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

# Optional: vector DB (Chroma)
try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None


def get_engine(db_path: str) -> Engine:
    """Return a SQLAlchemy engine for SQLite path."""
    # sqlite needs check_same_thread for some setups, but SQLAlchemy manages connections:
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, future=True)
    return engine


def init_db(engine: Engine) -> None:
    """Initialize DB pragmas and perform any initial migrations if required."""
    with engine.begin() as conn:
        # Use WAL for better concurrency
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        conn.exec_driver_sql("PRAGMA foreign_keys=ON;")


def query_sql(engine: Engine, sql: str) -> List[Dict[str, Any]]:
    """Execute SQL and return list of dict rows."""
    with engine.begin() as conn:
        res = conn.execute(text(sql))
        cols = res.keys()
        return [dict(zip(cols, row)) for row in res.fetchall()]


def safe_select(engine: Engine, sql: str) -> List[Dict[str, Any]]:
    """
    Very strict guard for SQL executed from LLM.
    Only allow single SELECT statements without semicolons or dangerous keywords.
    """
    if not isinstance(sql, str):
        raise ValueError("SQL must be a string.")
    s = sql.strip().lower()
    # disallow semicolons, multiple statements
    if ";" in s:
        raise ValueError("Multiple statements are not allowed.")
    # must start with select
    if not s.startswith("select "):
        raise ValueError("Only SELECT queries are allowed.")
    # deny dangerous keywords
    forbidden = [" drop ", " delete ", " update ", " insert ", " pragma ", " attach ", " alter ", " create "]
    for bad in forbidden:
        if bad in s:
            raise ValueError("Query contains forbidden operation.")
    # If it passes basic checks, run it
    return query_sql(engine, sql)


def summarize_schema(engine: Engine, dataset_id: str) -> Dict[str, Any]:
    """
    Inspect tables belonging to dataset_id and return structured schema info.
    Tables created by ingest are named <dataset_id>__<sheetname>.
    """
    insp = inspect(engine)
    all_tables = insp.get_table_names()
    tables = [t for t in all_tables if t.startswith(f"{dataset_id}__")]
    summary: Dict[str, Any] = {"dataset_id": dataset_id, "tables": []}
    with engine.begin() as conn:
        for t in tables:
            cols = []
            for col in insp.get_columns(t):
                cols.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True)
                })
            try:
                count = conn.execute(text(f'SELECT COUNT(*) as c FROM "{t}"')).scalar_one()
            except Exception:
                count = 0
            summary["tables"].append({"name": t, "row_count": int(count), "columns": cols})
    return summary


# --------------------- Chroma helpers (optional) ---------------------
_chroma_client = None


def _get_chroma_client():
    """Return a persistent chroma client if available, else None."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    if chromadb is None:
        return None
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")
    os.makedirs(data_dir, exist_ok=True)
    try:
        _chroma_client = chromadb.PersistentClient(path=data_dir)
    except Exception:
        _chroma_client = None
    return _chroma_client


def _get_embedding_fn():
    """Return an embedding function for Chroma, preferring cloud keys if present."""
    if embedding_functions is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            )
        except Exception:
            pass
    # fallback to sentence-transformers local model
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
        )
    except Exception:
        return None


def store_embeddings(dataset_id: str, embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
    client = _get_chroma_client()
    if client is None:
        return
    ef = _get_embedding_fn()
    collection = client.get_or_create_collection(name=dataset_id, embedding_function=ef)
    try:
        collection.add(embeddings=embeddings, metadatas=metadatas, ids=[str(i) for i in ids])
    except Exception:
        # don't fail ingestion if embeddings fail
        pass


def store_documents(dataset_id: str, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
    """Store raw texts so Chroma can compute embeddings if configured."""
    client = _get_chroma_client()
    if client is None:
        return
    ef = _get_embedding_fn()
    collection = client.get_or_create_collection(name=dataset_id, embedding_function=ef)
    try:
        collection.add(documents=documents, metadatas=metadatas, ids=[str(i) for i in ids])
    except Exception:
        pass


def query_embeddings(dataset_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    client = _get_chroma_client()
    if client is None:
        return []
    ef = _get_embedding_fn()
    collection = client.get_or_create_collection(name=dataset_id, embedding_function=ef)
    try:
        res = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    except Exception:
        return []
    results: List[Dict[str, Any]] = []
    ids = res.get("ids", [[]])[0]
    for i in range(len(ids)):
        results.append({
            "id": ids[i],
            "distance": res.get("distances", [[None]])[0][i],
            "metadata": res.get("metadatas", [[{}]])[0][i],
            "document": res.get("documents", [[None]])[0][i],
        })
    return results


def query_texts(dataset_id: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    client = _get_chroma_client()
    if client is None:
        return []
    ef = _get_embedding_fn()
    collection = client.get_or_create_collection(name=dataset_id, embedding_function=ef)
    try:
        res = collection.query(query_texts=[query_text], n_results=top_k)
    except Exception:
        return []
    results: List[Dict[str, Any]] = []
    ids = res.get("ids", [[]])[0]
    for i in range(len(ids)):
        results.append({
            "id": ids[i],
            "distance": res.get("distances", [[None]])[0][i],
            "metadata": res.get("metadatas", [[{}]])[0][i],
            "document": res.get("documents", [[None]])[0][i],
        })
    return results
