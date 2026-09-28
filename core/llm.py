"""Talk-to-your-data: natural language -> SQL (Gemini / Groq) with a rule-based offline fallback."""
import json, os, re, sqlite3, urllib.request
from functools import lru_cache
from .data import get_orders

SCHEMA = """Table orders(order_id INTEGER, date TEXT 'YYYY-MM-DD', customer_id INTEGER, region TEXT
(North/South/East/West), category TEXT (Electronics/Fashion/Grocery/Home/Sports), quantity INTEGER,
unit_price REAL, revenue REAL, profit REAL). Data covers 2024-01-01 to 2025-12-31."""

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|truncate|vacuum)\b", re.I)


@lru_cache(maxsize=1)
def _conn():
    con = sqlite3.connect(":memory:", check_same_thread=False)
    df = get_orders().copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_sql("orders", con, index=False)
    con.execute("PRAGMA query_only = ON")
    return con


def validate_sql(sql):
    sql = sql.strip().rstrip(";").strip()
    if not sql.lower().startswith(("select", "with")) or ";" in sql or FORBIDDEN.search(sql):
        raise ValueError("Only a single read-only SELECT query is allowed.")
    return sql


def _post(url, body, headers):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _prompt(q):
    return (f"You are a SQLite expert. {SCHEMA}\nWrite ONE SQLite SELECT query answering the question. "
            f"Return only the SQL, no markdown, no explanation.\nQuestion: {q}")


def _llm_sql(q):
    if os.getenv("GEMINI_API_KEY"):
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                  {"contents": [{"parts": [{"text": _prompt(q)}]}]},
                  {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        return r["candidates"][0]["content"]["parts"][0]["text"], "gemini"
    if os.getenv("GROQ_API_KEY"):
        r = _post("https://api.groq.com/openai/v1/chat/completions",
                  {"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                   "messages": [{"role": "user", "content": _prompt(q)}], "temperature": 0},
                  {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]})
        return r["choices"][0]["message"]["content"], "groq"
    return None, None


def _rules_sql(q):
    q = q.lower()
    asc = any(w in q for w in ("lowest", "least", "worst", "min", "bottom"))
    order = "ASC" if asc else "DESC"
    if "region" in q:
        col = "profit" if "profit" in q else "revenue"
        return f"SELECT region, ROUND(SUM({col}),2) AS {col} FROM orders GROUP BY region ORDER BY {col} {order}"
    if "categor" in q or "product" in q:
        col = "profit" if "profit" in q else "revenue"
        return f"SELECT category, ROUND(SUM({col}),2) AS {col} FROM orders GROUP BY category ORDER BY {col} {order}"
    if "customer" in q:
        return f"SELECT customer_id, ROUND(SUM(revenue),2) AS revenue FROM orders GROUP BY customer_id ORDER BY revenue {order} LIMIT 10"
    if "month" in q:
        return "SELECT strftime('%Y-%m', date) AS month, ROUND(SUM(revenue),2) AS revenue FROM orders GROUP BY month ORDER BY month"
    return "SELECT COUNT(*) AS orders, ROUND(SUM(revenue),2) AS revenue, ROUND(SUM(profit),2) AS profit FROM orders"


def ask(question):
    engine, note = "rules", None
    try:
        raw, eng = _llm_sql(question)
        if raw:
            sql = re.sub(r"```(?:sql)?|```", "", raw).strip()
            engine = eng
        else:
            sql = _rules_sql(question)
            note = "No LLM key found - used offline rule-based SQL. Add GEMINI_API_KEY or GROQ_API_KEY for full AI."
    except Exception as e:  # network / quota errors -> fall back gracefully
        sql, note = _rules_sql(question), f"LLM unavailable ({type(e).__name__}); used rule-based fallback."
    sql = validate_sql(sql)
    cur = _conn().execute(sql)
    cols = [c[0] for c in cur.description]
    rows = [list(r) for r in cur.fetchmany(100)]
    return {"question": question, "sql": sql, "engine": engine, "note": note, "columns": cols, "rows": rows}


def sentiment(text):
    from .analytics import lexicon_sentiment
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            url = os.getenv("HF_SENTIMENT_URL", "https://router.huggingface.co/hf-inference/models/"
                            "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
            r = _post(url, {"inputs": text}, {"Authorization": "Bearer " + token})
            top = max(r[0] if isinstance(r[0], list) else r, key=lambda x: x["score"])
            return {"label": top["label"], "score": round(top["score"], 3), "engine": "huggingface"}
        except Exception:
            pass
    return lexicon_sentiment(text)
