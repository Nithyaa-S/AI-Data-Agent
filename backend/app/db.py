from __future__ import annotations
import os
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text, inspect, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from .config import settings

# Optional: vector DB (Chroma)
try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None

# Database session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False)

# Logger
logger = logging.getLogger(__name__)

def get_engine(database_url: str = None) -> Engine:
    """Return a SQLAlchemy engine for the given database URL.
    
    Args:
        database_url: The database URL. If None, uses settings.DATABASE_URL
    """
    if database_url is None:
        database_url = settings.DATABASE_URL

    if not database_url:
        raise ValueError("DATABASE_URL is required")

    # SQLite configuration (Windows-safe)
    if database_url.startswith('sqlite'):
        # Ensure directory exists for file-based SQLite
        if database_url.startswith('sqlite:///'):
            db_path = database_url.replace('sqlite:///', '', 1)
            if not db_path.startswith(':memory:'):
                db_dir = os.path.dirname(db_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)

        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
            future=True,
        )

        @event.listens_for(engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        # PostgreSQL or others
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            future=True,
        )

    return engine


def init_db(engine: Engine) -> None:
    """Initialize DB pragmas and perform any initial migrations if required.
    Only run PRAGMA statements for SQLite engines.
    """
    try:
        if engine.url.drivername.startswith("sqlite"):
            with engine.begin() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                conn.exec_driver_sql("PRAGMA foreign_keys=ON;")
        else:
            # No-op for Postgres/others
            return
    except Exception as e:
        logger.warning(f"init_db skipped pragmas: {e}")


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

def _get_table_row_count(engine: Engine, table_name: str) -> int:
    """Return COUNT(*) for a table; 0 on error."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
            val = result.scalar()
            return int(val or 0)
    except Exception as e:
        logger.warning(f"Could not get row count for {table_name}: {e}")
        return 0
def summarize_schema(engine: Engine, dataset_id: str) -> Dict[str, Any]:
    """Inspect tables for the given dataset and return structured schema info."""
    insp = inspect(engine)
    is_postgres = engine.url.drivername == 'postgresql'

    # Discover tables
    if is_postgres:
        with engine.connect() as conn:
            result = conn.execute(text(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND LOWER(table_name) LIKE :prefix
                """
            ), {"prefix": f"{dataset_id.lower()}%"})
            all_tables = [r[0] for r in result]
    else:
        all_tables = insp.get_table_names()

    # Accept both "<id>__<sheet>" and "<id>_<sheet>" (in case of slug collapse)
    def _match_tables(id_str: str) -> List[str]:
        d = id_str.lower()
        pref_double = f"{d}__"
        pref_single = f"{d}_"
        return [t for t in all_tables if t.lower().startswith(pref_double) or t.lower().startswith(pref_single)]

    tables = _match_tables(dataset_id)
    # Fallback: if caller passed bare hex without 'ds_' prefix, try with 'ds_' prefix
    if not tables and not dataset_id.lower().startswith('ds_'):
        tables = _match_tables(f"ds_{dataset_id}")

    logger.info(f"All tables in database: {all_tables}")
    logger.info(f"Filtered tables for dataset '{dataset_id}': {tables}")

    schema: Dict[str, Any] = {"tables": [], "database_type": engine.url.drivername}

    for t in tables:
        columns: List[Dict[str, Any]] = []
        try:
            cols = insp.get_columns(t)
            for col in cols:
                columns.append({
                    "name": col.get("name"),
                    "type": str(col.get("type")),
                    "nullable": col.get("nullable", True),
                    "default": str(col.get("default")) if col.get("default") is not None else None,
                })

            # Sample rows
            sample: List[Dict[str, Any]] = []
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f'SELECT * FROM "{t}" LIMIT 5'))
                    sample = [dict(row._mapping) for row in result]
            except Exception as e:
                logger.warning(f"Could not sample {t}: {e}")

            # Row count
            count = _get_table_row_count(engine, t)

            schema["tables"].append({
                "name": t,
                "columns": columns,
                "sample": sample,
                "row_count": count,
            })
        except Exception as e:
            logger.error(f"Error processing table {t}: {e}")
            continue

    return schema


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
    # Allow disabling embeddings entirely in production to avoid heavy model downloads
    if os.getenv("DISABLE_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
        return None
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
