# Cordly AI Backend (FastAPI)

## Setup

1. Create a virtualenv and install deps

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Configure LLM (optional but recommended)

- Set `OPENAI_API_KEY` in your environment.
- Optionally set `OPENAI_MODEL` (default: gpt-4o-mini)

## Endpoints

- GET `/health`
- POST `/api/upload` (multipart/form-data) with `file` as Excel
- GET `/api/schema/{dataset_id}`
- POST `/api/ask` with JSON `{ dataset_id, question }`
