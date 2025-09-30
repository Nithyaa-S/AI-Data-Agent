from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import logging

from .db import init_db, get_engine, summarize_schema, query_sql
from .ingest import ingest_excel
from .nlp import NLQueryEngine, answer_query

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cordly_ai")

app = FastAPI(title="Cordly AI - Excel Conversational Analytics", version="0.2.0")

# CORS - include production domains (no wildcard in allow_origins; use regex instead)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://agentdata.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.netlify\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load settings from environment
try:
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        DATABASE_URL: str | None = None
        GROQ_API_KEY: str | None = None
        GROQ_MODEL: str = "llama-3.3-70b-versatile"
        OPENAI_API_KEY: str | None = None
        OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
        SENTENCE_TRANSFORMERS_MODEL: str = "all-MiniLM-L6-v2"

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

    _settings = Settings()
    for k, v in _settings.model_dump().items():
        if v is not None and os.getenv(k) is None:
            os.environ[k] = str(v)
except Exception as e:
    logger.warning(f"Could not load settings: {e}")

# Get database URL from environment or use SQLite fallback
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Fallback to SQLite for local development
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    DB_PATH = os.path.join(DATA_DIR, "app.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    logger.info(f"Using SQLite database at: {DB_PATH}")
else:
    logger.info(f"Using database from DATABASE_URL: {DATABASE_URL[:20]}...")

# DB engine & init
try:
    engine = get_engine(DATABASE_URL)
    init_db(engine)
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

# NL engine
nl_engine = NLQueryEngine()


# Response models -----------------------------------------------------
class UploadResponse(BaseModel):
    dataset_id: str
    sheets: List[str]
    schema: Dict[str, Any]
    samples: Dict[str, List[Dict[str, Any]]]


class AskRequest(BaseModel):
    dataset_id: str
    question: str
    prefer_table: Optional[bool] = None


class AskResponse(BaseModel):
    answer: str
    sql: Optional[str] = None
    table: Optional[Dict[str, Any]] = None
    chart: Optional[Dict[str, Any]] = None


# Endpoints ----------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Cordly AI API is running", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_excel(file: UploadFile = File(...)):
    # Validate extension loosely
    if not file.filename.lower().endswith((".xlsx", ".xls", ".xlsb", ".csv")):
        raise HTTPException(status_code=400, detail="Only Excel (.xlsx/.xls/.xlsb) or CSV is supported")

    content = await file.read()
    logger.info(f"Starting upload for file: {file.filename}")
    
    dataset_id, sheet_names = ingest_excel(engine, file.filename, content)
    logger.info(f"Ingestion complete. Dataset ID: {dataset_id}, Sheets: {sheet_names}")

    schema = summarize_schema(engine, dataset_id)
    logger.info(f"Schema summary: {len(schema.get('tables', []))} tables found")
    
    samples: Dict[str, List[Dict[str, Any]]] = {}
    for sheet in sheet_names:
        table_name = f"{dataset_id}__{sheet}"
        try:
            rows = query_sql(engine, f'SELECT * FROM "{table_name}" LIMIT 10')
            logger.info(f"Fetched {len(rows)} sample rows for sheet '{sheet}' (table: {table_name})")
        except Exception as e:
            logger.error(f"Error fetching samples for {table_name}: {e}")
            rows = []
        samples[sheet] = rows

    logger.info(f"Upload complete. Returning {len(sheet_names)} sheets with {sum(len(v) for v in samples.values())} total sample rows")
    return UploadResponse(dataset_id=dataset_id, sheets=sheet_names, schema=schema, samples=samples)


@app.get("/api/schema/{dataset_id}")
async def get_schema(dataset_id: str):
    try:
        return summarize_schema(engine, dataset_id)
    except Exception as e:
        logger.exception("schema error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(req: AskRequest, request: Request):
    """Answer a user's question against an uploaded dataset."""
    logger.info("Ask request for dataset %s: %s", req.dataset_id, (req.question or "")[:120])
    # Validate dataset exists
    try:
        dsid = req.dataset_id or ""
        schema = summarize_schema(engine, dsid)
        # Fallback: allow bare ids without 'ds_' prefix
        if not schema.get("tables") and not dsid.lower().startswith("ds_"):
            dsid_pref = f"ds_{dsid}"
            schema = summarize_schema(engine, dsid_pref)
            if schema.get("tables"):
                dsid = dsid_pref
        if not schema.get("tables"):
            raise HTTPException(status_code=404, detail="Dataset not found or has no tables")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to fetch schema")
        raise HTTPException(status_code=500, detail="Failed to fetch dataset schema")

    # Primary attempt using enhanced answer_query
    try:
        result = answer_query(engine, dsid, req.question)
        # Safely coerce response to fit AskResponse model
        return AskResponse(
            answer=result.get("answer", ""),
            sql=result.get("sql"),
            table=result.get("table"),
            chart=result.get("chart"),
        )
    except Exception as e:
        logger.exception("Primary answer_query failed, falling back to NL plan")
        # Fallback: use NLQueryEngine plan & optionally execute
        try:
            plan = nl_engine.plan(req.question, schema)
            data = []
            if plan.get("sql") and plan.get("execute", True):
                try:
                    data = query_sql(engine, plan["sql"])
                    plan["result_preview"] = data[:50]
                except Exception as ex:
                    plan["error"] = str(ex)
            return AskResponse(
                answer=plan.get("answer", ""),
                sql=plan.get("sql"),
                table={"columns": list(data[0].keys()) if data else [], "rows": data} if plan.get("sql") else None,
                chart=plan.get("chart"),
            )
        except Exception as e2:
            logger.exception("Fallback also failed")
            raise HTTPException(status_code=500, detail="Unable to compute answer")