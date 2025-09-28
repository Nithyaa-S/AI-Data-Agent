from __future__ import annotations
import os
import json
import re
from typing import Dict, Any, List, Tuple
import datetime as dt

from .db import query_sql, summarize_schema, query_texts, safe_select

# Try to import Groq client if available
try:
    from groq import Groq
except Exception:
    Groq = None

SYSTEM_PROMPT = (
    "You are a world-class data analyst. Given a SQLite schema and a user question, "
    "produce: (1) a concise, helpful natural language answer plan, (2) an executable SQLite SQL query if appropriate, "
    "and (3) an optional simple chart spec.\n\n"
    "Chart spec format: {type: 'bar'|'line'|'pie'|'scatter', x: <column>, y: <column or aggregation>, series?: <column>, title?: str}.\n"
    "Only output valid JSON with keys: answer, sql, chart, execute.\n"
    "If you cannot produce a SQL query, set execute=false and return a helpful answer."
)


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first top-level JSON object from an LLM response.
    Uses a simple brace-balancing scan (no recursive regex).
    If extraction fails, return {"answer": text} as a fallback.
    """
    if not isinstance(text, str):
        return {"answer": str(text)}
    s = text.strip()
    start = s.find("{")
    if start == -1:
        return {"answer": s}
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = s[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break
    # fallback: try naive slice between first '{' and last '}'
    end = s.rfind('}')
    if end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return {"answer": s}


def _is_simple_aggregation(question: str) -> bool:
    q = (question or "").lower()
    keywords = [
        "count", "sum", "average", "avg", "min", "max",
        "top", "bottom", "group", "trend", "by ",
        "how many", "number of", "total ", "total number"
    ]
    return any(k in q for k in keywords)


def _package_chart_data(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """
    If rows have exactly 2 columns and numeric y, produce chart_data list and chart_type.
    """
    if not rows:
        return [], ""
    cols = list(rows[0].keys())
    if len(cols) != 2:
        return [], ""
    x_col, y_col = cols[0], cols[1]

    def is_date(val):
        if val is None:
            return False
        if isinstance(val, (dt.date, dt.datetime)):
            return True
        try:
            dt.datetime.fromisoformat(str(val))
            return True
        except Exception:
            return False

    def is_number(val):
        try:
            float(val)
            return True
        except Exception:
            return False

    # check a few rows for types
    x_is_date = any(is_date(r[x_col]) for r in rows[:10] if x_col in r)
    y_is_num = any(is_number(r[y_col]) for r in rows[:10] if y_col in r)
    if not y_is_num:
        return [], ""
    chart_data = []
    for r in rows:
        try:
            y_raw = r.get(y_col)
            y_val = float(y_raw) if y_raw not in (None, "") else None
            chart_data.append({"x": r.get(x_col), "y": y_val})
        except Exception:
            continue
    chart_type = "line" if x_is_date else "bar"
    return chart_data, chart_type


class NLQueryEngine:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        self.enabled = bool(api_key and Groq)
        try:
            self.client = Groq(api_key=api_key) if self.enabled else None
        except Exception:
            self.client = None
            self.enabled = False
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    def plan(self, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Return a plan (rule-based if LLM disabled)."""
        if self.enabled and self.client:
            return self._plan_llm(question, schema)
        return self._plan_rule_based(question, schema)

    def _plan_llm(self, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        schema_str = json.dumps(schema, indent=2)
        prompt = f"Schema:\n{schema_str}\n\nQuestion: {question}\nRespond with JSON as specified."
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            # Groq-style response shape handling
            text = ""
            try:
                text = resp.choices[0].message.content
            except Exception:
                # fallback if structure different
                text = getattr(resp, "text", str(resp))
            plan = _extract_json(text)
            # ensure keys exist
            plan.setdefault("answer", plan.get("answer", ""))
            plan.setdefault("sql", plan.get("sql"))
            plan.setdefault("execute", bool(plan.get("sql")))
            plan.setdefault("chart", plan.get("chart"))
            return plan
        except Exception as e:
            return {"answer": f"LLM planning error: {e}", "execute": False}

    def _plan_rule_based(self, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        tables = schema.get("tables", [])
        if not tables:
            return {"answer": "No tables found in the dataset.", "execute": False}
        table_meta = tables[0]
        table = table_meta["name"]
        cols = [c["name"] for c in table_meta.get("columns", [])]
        # a helpful fallback: suggest common queries
        sample_cols = ", ".join(cols[:5]) if cols else ""
        answer = (
            f"No LLM configured. I can run basic queries on table '{table}'. "
            f"Available columns: {sample_cols}."
        )
        sql = f'SELECT COUNT(*) AS row_count FROM "{table}"'
        chart = {"type": "bar", "x": "metric", "y": "value", "title": "Basic Row Count"}
        return {"answer": answer, "sql": sql, "chart": chart, "execute": True}


# ---------------- Enhanced LLM Answering / Execution ----------------

def _norm(s: str) -> str:
    """Normalize a string for loose matching: lowercase and remove non-alnum."""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _first_table_and_columns(schema: Dict[str, Any]) -> Tuple[str, List[str]]:
    tables = schema.get("tables", [])
    if not tables:
        return "", []
    t = tables[0]
    table = t["name"]
    cols = [c["name"] for c in t.get("columns", [])]
    return table, cols


def _simple_filter_sql(question: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Very small rule-based detector for exact-id or region/state filtering.
    Returns (sql, extras) or ("", {}). Only targets the first table for simplicity.
    Handles:
      - "details of 2605000446 orderid" -> WHERE orderid = 2605000446
      - "List of products ordered from Madhya Pradesh" -> WHERE <state|region|...> LIKE '%Madhya Pradesh%'
    """
    if not question:
        return "", {}
    table, cols = _first_table_and_columns(schema)
    if not table or not cols:
        return "", {}

    norm_map = {_norm(c): c for c in cols}
    ql = question.strip()

    # 1) Exact-id pattern: capture a large integer and a nearby column token present in cols
    # Example: "details of 2605000446 orderid" or "orderid 2605000446"
    tokens = re.findall(r"[\w'-]+", ql)
    numbers = [t for t in tokens if t.isdigit()]
    if numbers:
        # search for a column-like token in the same question
        for t in tokens:
            n = _norm(t)
            if n in norm_map:
                col = norm_map[n]
                # Use the first number hit
                val = numbers[0]
                # Prefer quoting column and using parameterless safe_select (already sanitized by safe_select)
                sql = f'SELECT * FROM "{table}" WHERE "{col}" = {val} LIMIT 200'
                return sql, {"detected": "id_filter", "column": col, "value": val}

    # 2) Region/state-like filters: look for quoted or title-cased multiword (e.g., Madhya Pradesh)
    # We will try to find a likely geo column in the schema first
    geo_hints = ["state", "region", "province", "location", "city"]
    geo_cols = [c for c in cols if any(h in _norm(c) for h in geo_hints)]
    if geo_cols:
        # Try to extract a location phrase
        # Heuristic: longest capitalized phrase or quoted string
        m = re.search(r'"([^"]+)"|\'([^\']+)\'', ql)
        place = None
        if m:
            place = m.group(1) or m.group(2)
        else:
            # find consecutive TitleCase tokens
            groups: List[str] = []
            cur: List[str] = []
            for t in tokens:
                if len(t) > 2 and t[0].isupper() and any(ch.islower() for ch in t[1:]):
                    cur.append(t)
                else:
                    if cur:
                        groups.append(" ".join(cur))
                        cur = []
            if cur:
                groups.append(" ".join(cur))
            place = max(groups, key=len) if groups else None
        if place:
            col = geo_cols[0]
            sql = f'SELECT * FROM "{table}" WHERE "{col}" LIKE "%{place}%" LIMIT 200'
            return sql, {"detected": "geo_filter", "column": col, "value": place}

    return "", {}

def answer_query(engine, dataset_id: str, question: str) -> Dict[str, Any]:
    """
    High-level QA/SQL generation + execution. Uses Groq LLM when available.
    Returns a dict with keys: answer, sql (optional), table (optional), chart (optional).
    """
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if (api_key and Groq) else None
    model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    schema = summarize_schema(engine, dataset_id)

    # First, try a deterministic rule-based filter for common intents (id/geo filters)
    simple_sql, meta = _simple_filter_sql(question, schema)
    if simple_sql:
        try:
            rows = safe_select(engine, simple_sql)
            chart_data, chart_type = _package_chart_data(rows)
            result: Dict[str, Any] = {
                "answer": "Here are the results.",
                "sql": simple_sql,
            }
            if rows:
                result["table"] = {"columns": list(rows[0].keys()), "rows": rows}
            if chart_data:
                result["chart"] = {"type": chart_type, "chart_data": chart_data}
            return result
        except Exception:
            # fall through to LLM
            pass

    # If LLM is available, NEXT try to produce a single safe SQL for any question
    if client is not None:
        prompt = (
            "You are a data analyst working with SQLite. Given a schema and a user question, "
            "write a single safe SQLite SELECT statement that answers the question.\n"
            "- Do not modify data.\n"
            "- Prefer GROUP BY for aggregations.\n"
            "- Use correct table and column names from the schema.\n"
            "Return only JSON: {sql: \"...\"}."
        )

        # RAG context from Chroma - best-effort
        rag = []
        try:
            rag = query_texts(dataset_id, question, top_k=5)
        except Exception:
            rag = []

        ctx = {"schema": schema, "question": question, "context": rag}
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(ctx)},
                ],
                temperature=0.1,
            )
            text = ""
            try:
                text = resp.choices[0].message.content
            except Exception:
                text = getattr(resp, "text", str(resp))
            sql_json = _extract_json(text)
            sql = sql_json.get("sql")
            if not sql:
                # fall through to QA if no SQL
                pass
            # Execute safely
            rows: List[Dict[str, Any]] = []
            if sql:
                try:
                    rows = safe_select(engine, sql)
                    chart_data, chart_type = _package_chart_data(rows)
                    result: Dict[str, Any] = {"answer": "Here are the results.", "sql": sql}
                    if rows:
                        result["table"] = {"columns": list(rows[0].keys()), "rows": rows}
                    if chart_data:
                        result["chart"] = {"type": chart_type, "chart_data": chart_data}
                    return result
                except Exception as e:
                    # If SQL invalid/unsafe, continue to QA path
                    pass
        except Exception as e:
            return {"answer": f"LLM SQL error: {e}"}

    # If no client or SQL path didn't yield results, do a QA-style or fallback
    if client is None:
        # helpful fallback: show available tables/columns and sample rows
        tables = schema.get("tables", [])
        if not tables:
            return {"answer": "No tables found in the dataset."}
        table_meta = tables[0]
        table = table_meta["name"]
        cols = [c["name"] for c in table_meta.get("columns", [])]
        try:
            rows = safe_select(engine, f'SELECT * FROM "{table}" LIMIT 10')
        except Exception:
            rows = []
        return {
            "answer": (
                f"LLM not configured. Table '{table}' has {table_meta.get('row_count', 0)} rows. "
                f"Available columns: {', '.join(cols[:10])}. Try asking: 'What is the average of <column>' or "
                "'Show trend of <column> by <date_column>'."
            ),
            "table": {"columns": list(rows[0].keys()) if rows else [], "rows": rows},
        }

    # When LLM present but SQL path didn't produce results, use LLM for QA
    qa_prompt = (
        "You answer questions about tabular data. Use the provided schema to answer concisely.\n"
        "If a calculation is required, reason step-by-step but return only the final answer.\n"
        "Return JSON: {answer: \"...\"}."
    )
    rag = []
    try:
        rag = query_texts(dataset_id, question, top_k=5)
    except Exception:
        rag = []
    ctx = {"schema": schema, "question": question, "context": rag}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": qa_prompt},
                {"role": "user", "content": json.dumps(ctx)},
            ],
            temperature=0.2,
        )
        text = ""
        try:
            text = resp.choices[0].message.content
        except Exception:
            text = getattr(resp, "text", str(resp))
        ans = _extract_json(text)
        return {"answer": ans.get("answer", text)}
    except Exception as e:
        return {"answer": f"LLM QA error: {e}"}
