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


def _select_best_table(schema: Dict[str, Any], question: str) -> Tuple[str, List[str]]:
    """Pick the table whose columns best match the question terms.
    If no overlap, return the first table.
    """
    tables = schema.get("tables", [])
    if not tables:
        return "", []
    # tokenize question
    q = (question or "").lower()
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    best = None
    best_score = -1
    for t in tables:
        cols = [c["name"] for c in t.get("columns", [])]
        score = 0
        for c in cols:
            cn = _norm(c)
            # increment if any token is substring of column name
            score += sum(1 for tok in q_tokens if tok and tok in cn)
        # small bonus if table name matches
        if any(tok in _norm(t["name"]) for tok in q_tokens):
            score += 1
        if score > best_score:
            best = t
            best_score = score
    target = best or tables[0]
    return target["name"], [c["name"] for c in target.get("columns", [])]


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
    table, cols = _select_best_table(schema, question)
    if not table or not cols:
        return "", {}

    q = question.strip()
    ql = q.lower()

    # 1) Try to detect an ID value (numeric or quoted string)
    id_value: str | None = None
    id_is_string = False
    # quoted token "..."
    m_q = re.search(r'"([^"]+)"', q) or re.search(r"'([^']+)'", q)
    if m_q:
        id_value = m_q.group(1)
        id_is_string = True
    else:
        # bare number
        m_n = re.search(r"\b(\d{2,})\b", q)  # 2+ digit number
        if m_n:
            id_value = m_n.group(1)

    # Candidate id columns ranked by relevance
    id_col_hints = [
        "orderid", "order_id", "order id",
        "customerid", "customer_id", "customer id",
        "productid", "product_id", "product id",
        "id"
    ]
    norm_cols = { _norm(c): c for c in cols }

    def pick_best_id_column() -> str:
        # prefer an id column whose hint appears in the question, else general 'id'
        for hint in id_col_hints:
            if hint in ql:
                # find closest column name
                for k, orig in norm_cols.items():
                    if hint.replace(" ", "") in k:
                        return orig
        # fallback: any column that ends with 'id'
        for k, orig in norm_cols.items():
            if k.endswith("id"):
                return orig
        return ""

    # If we have an id value and some id-like column, build a simple filter
    if id_value is not None:
        col = pick_best_id_column()
        if col:
            if id_is_string:
                sql = f'SELECT * FROM "{table}" WHERE "{col}" = :val LIMIT 50'
                # use literal formatting because safe_select doesn't accept params; sanitize quotes
                safe_val = id_value.replace("\"", "").replace("'", "")
                sql = sql.replace(":val", f"'{safe_val}'")
            else:
                sql = f'SELECT * FROM "{table}" WHERE "{col}" = {id_value} LIMIT 50'
            return sql, {"detected": "id_filter", "column": col, "value": id_value}

    # 2) Geo/text contains filter (state/region/city/country)
    geo_hints = ["state", "region", "city", "country", "area", "province", "location"]
    geo_value = None
    # try to grab last capitalized phrase as location-like token
    tokens = re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*", q)
    if tokens:
        geo_value = tokens[-1]
    if geo_value:
        for c in cols:
            cn = _norm(c)
            if any(h in cn for h in geo_hints):
                like = geo_value.replace("'", "")
                # case-insensitive across SQLite/Postgres
                sql = (
                    f'SELECT * FROM "{table}" '
                    f'WHERE LOWER("{c}") LIKE LOWER('\'%{like}%\'') LIMIT 100'
                )
                return sql, {"detected": "geo_filter", "column": c, "value": geo_value}

    return "", {}

def _calculate_total_revenue(question: str, table: str, cols: List[str]) -> Tuple[str, Dict[str, Any]]:
    """Handle total revenue calculation by multiplying price and quantity columns."""
    q = question.lower()
    if "total revenue" not in q and "total sales" not in q and not ("price" in q and "quantity" in q):
        return "", {}
        
    price_col = next((c for c in cols if "price" in c.lower()), None)
    qty_col = next((c for c in cols if any(x in c.lower() for x in ["quantity", "qty"])), None)
    
    if not price_col or not qty_col:
        return "", {}
        
    sql = f"""
    SELECT 
        SUM(CAST(REPLACE(REPLACE("{price_col}", '$', ''), ',', '') AS REAL) * 
             CAST(REPLACE(REPLACE("{qty_col}", '$', ''), ',', '') AS REAL)) 
        AS total_revenue 
    FROM "{table}"
    """
    
    return sql.strip(), {"detected": "total_revenue", "price_col": price_col, "qty_col": qty_col}

def _top_n_sql(question: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Detect patterns like "top 5 <entities> by <metric>" or "bottom 10 ... by ..." and build a SQL with
    GROUP BY <entity> and ORDER BY aggregated <metric> with LIMIT N.
    Only targets the first table for simplicity.
    """
    if not question:
        return "", {}
        
    table, cols = _select_best_table(schema, question)
    if not table or not cols:
        return "", {}

    # First check if this is a total revenue calculation
    rev_sql, rev_meta = _calculate_total_revenue(question, table, cols)
    if rev_sql:
        return rev_sql, rev_meta

    q = question.lower()
    m = re.search(r"\b(top|bottom)\s+(\d+)\b", q)
    if not m:
        return "", {}
    direction, n_str = m.group(1), m.group(2)
    try:
        n = max(1, min(1000, int(n_str)))
    except Exception:
        n = 5

    # Try to find metric after "by <metric>"
    metric_match = re.search(r"\bby\s+([a-zA-Z0-9_ ]+)", question, re.IGNORECASE)
    metric_raw = metric_match.group(1).strip() if metric_match else ""

    norm_map = {_norm(c): c for c in cols}
    metric_col = None
    if metric_raw:
        # choose the column in schema that best matches metric_raw
        mr = _norm(metric_raw)
        # look for exact or substring matches
        for k, orig in norm_map.items():
            if mr == k or mr in k or k in mr:
                metric_col = orig
                break
    # fallback: common metric names
    if not metric_col:
        for hint in ["profit", "revenue", "sales", "amount", "quantity"]:
            for c in cols:
                if hint in _norm(c):
                    metric_col = c
                    break
            if metric_col:
                break
    if not metric_col:
        return "", {}

    # pick an entity column to group by: prefer product/item/category/customer
    group_col = None
    for hint in ["product", "item", "category", "customer", "region", "name"]:
        for c in cols:
            if hint in _norm(c) and c != metric_col:
                group_col = c
                break
        if group_col:
            break
    # fallback: first non-metric text-like column
    if not group_col:
        for c in cols:
            if c != metric_col and not ("id" in _norm(c)):
                group_col = c
                break
    if not group_col:
        return "", {}

    agg = f'SUM("{metric_col}")'
    order = "ASC" if direction == "bottom" else "DESC"
    sql = (
        f'SELECT "{group_col}" AS label, {agg} AS value '\
        f'FROM "{table}" GROUP BY "{group_col}" ORDER BY value {order} LIMIT {n}'
    )
    return sql, {"detected": "top_n", "n": n, "metric": metric_col, "group": group_col, "direction": direction}

    return "", {}

def _max_min_row_sql(question: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Detect questions like 'which order has the highest price' or 'lowest profit row'.
    Returns a SELECT * ORDER BY <metric> DESC/ASC LIMIT 1.
    """
    if not question:
        return "", {}
    table, cols = _first_table_and_columns(schema)
    if not table or not cols:
        return "", {}
    q = question.lower()
    direction = None
    if any(k in q for k in ["highest", "max", "maximum", "top"]):
        direction = "DESC"
    elif any(k in q for k in ["lowest", "min", "minimum", "bottom"]):
        direction = "ASC"
    if not direction:
        return "", {}
    # detect metric
    norm_cols = {_norm(c): c for c in cols}
    metric_hints = ["price", "profit", "revenue", "amount", "quantity", "total"]
    metric_col = None
    for hint in metric_hints:
        for k, orig in norm_cols.items():
            if hint in k:
                metric_col = orig
                break
        if metric_col:
            break
    if not metric_col:
        return "", {}
    sql = f'SELECT * FROM "{table}" ORDER BY "{metric_col}" {direction} LIMIT 1'
    return sql, {"detected": "max_min_row", "metric": metric_col, "direction": direction}

def _equality_filter_sql(question: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Detect simple equality filters like product/customer/category equals a given value.
    Examples: 'orders for customer "Alice"', 'products of category Electronics'
    """
    if not question:
        return "", {}
    table, cols = _first_table_and_columns(schema)
    if not table or not cols:
        return "", {}
    q = question.strip()
    value = None
    m_q = re.search(r'"([^"]+)"', q) or re.search(r"'([^']+)'", q)
    if m_q:
        value = m_q.group(1)
    # entity columns to try
    entity_hints = ["product", "customer", "category", "name", "region", "city"]
    targets = []
    for c in cols:
        cn = _norm(c)
        if any(h in cn for h in entity_hints):
            targets.append(c)
    if not targets or not value:
        return "", {}
    safe_val = (value or "").replace("'", "")
    # Try first matching column
    col = targets[0]
    # Case-insensitive match for both SQLite and Postgres using LOWER()
    sql = (
        f'SELECT * FROM "{table}" '
        f'WHERE LOWER("{col}") LIKE LOWER('\'%{safe_val}%\'') LIMIT 100'
    )
    return sql, {"detected": "equality_filter", "column": col, "value": value}

def answer_query(engine, dataset_id: str, question: str) -> Dict[str, Any]:
    """
    High-level QA/SQL generation + execution. Uses Groq LLM when available.
    Returns a dict with keys: answer, sql (optional), table (optional), chart (optional).
    """
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if (api_key and Groq) else None
    model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    schema = summarize_schema(engine, dataset_id)

    # First, try ranking detector: top/bottom N by <metric>
    top_sql, top_meta = _top_n_sql(question, schema)
    if top_sql:
        try:
            rows = safe_select(engine, top_sql)
            chart_data, chart_type = _package_chart_data(rows)
            result: Dict[str, Any] = {
                "answer": "Here are the results.",
                "sql": top_sql,
            }
            if rows:
                result["table"] = {"columns": list(rows[0].keys()), "rows": rows}
            if chart_data:
                result["chart"] = {"type": chart_type or "bar", "chart_data": chart_data}
            return result
        except Exception:
            pass

    # Next, try a deterministic rule-based filter for common intents (id/geo filters)
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

    # Try max/min row by metric
    mm_sql, mm_meta = _max_min_row_sql(question, schema)
    if mm_sql:
        try:
            rows = safe_select(engine, mm_sql)
            result: Dict[str, Any] = {"answer": "Here are the results.", "sql": mm_sql}
            if rows:
                result["table"] = {"columns": list(rows[0].keys()), "rows": rows}
            return result
        except Exception:
            pass

    # Try generic equality filter on common entity columns
    eq_sql, eq_meta = _equality_filter_sql(question, schema)
    if eq_sql:
        try:
            rows = safe_select(engine, eq_sql)
            result: Dict[str, Any] = {"answer": "Here are the results.", "sql": eq_sql}
            if rows:
                result["table"] = {"columns": list(rows[0].keys()), "rows": rows}
            return result
        except Exception:
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
