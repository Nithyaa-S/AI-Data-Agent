# Cordly AI — Conversational Analytics for Excel

A full-stack platform where users upload any Excel file and ask complex business questions in natural language. The system cleans and ingests the data, understands questions with GenAI, and returns answers with relevant tables and charts.

- Frontend: React (Vite)
- Backend: FastAPI (Python)
- Database: SQLite (via SQLAlchemy)

## Features

- Robust Excel ingestion: handles unnamed columns, messy headers, multiple sheets, and mixed types.
- Automatic schema detection and sampling.
- Natural language Q&A with rule-based fallback if no LLM key.
- SQL execution with result tables and chart suggestions.

## Quickstart

### 1) Backend

```powershell
# In a terminal at: Cordly AI/backend
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (Optional) Create .env and set OpenAI key for best results
# copy .env.example to .env and fill OPENAI_API_KEY

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at http://localhost:8000

### 2) Frontend

```powershell
# In another terminal at: Cordly AI/frontend
npm install
# (Optional) copy .env.example to .env and adjust VITE_API_URL if needed
npm run dev
```

App runs at http://localhost:5173

## Using the App

1. Upload an Excel file (.xlsx, .xls, .xlsb) using the upload card.
2. Inspect detected sheets, tables, and sample rows.
3. Ask a question in natural language. If an OpenAI key is configured, the system will produce a sophisticated plan, SQL, and chart. Without a key, it returns a helpful fallback answer with a basic query.

## Configuration

- Backend uses environment variables:
  - `OPENAI_API_KEY` (optional but recommended)
  - `OPENAI_MODEL` (default: gpt-4o-mini)
- Frontend:
  - `VITE_API_URL` (default: http://localhost:8000)

## Deployment

- Backend: Deploy to Render/Railway/Fly.io with Python 3.11, run command:
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Frontend: Deploy to Netlify/Vercel. Build: `npm run build`, output: `dist/`.
- Ensure CORS allows the frontend URL in `backend/app/main.py`.

## Roadmap

- Authentication & multi-user datasets
- Semantic aggregation & multi-table reasoning
- Persistent chat sessions with citations
- Advanced charting and dashboarding

## License

MIT
