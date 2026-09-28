import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core import analytics as A, llm
from core.data import get_orders

app = FastAPI(title="AI Decision Intelligence Platform", version="1.0.0")


class Question(BaseModel):
    question: str


class Text(BaseModel):
    text: str


@app.get("/api/health")
def health():
    return {"status": "ok", "llm": "gemini" if os.getenv("GEMINI_API_KEY") else
            "groq" if os.getenv("GROQ_API_KEY") else "offline-rules"}


@app.get("/api/kpis")
def kpis():
    return A.kpis(get_orders())


@app.get("/api/forecast")
def forecast(horizon: int = 30):
    return A.forecast(get_orders(), horizon)


@app.get("/api/anomalies")
def anomalies():
    return A.anomalies(get_orders())


@app.get("/api/segments")
def segments():
    return A.rfm(get_orders())


@app.get("/api/churn")
def churn():
    return A.churn(get_orders())


@app.post("/api/ask")
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(400, "Question is empty")
    try:
        return llm.ask(body.question.strip()[:500])
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Query failed: {e}")


@app.post("/api/sentiment")
def sentiment(body: Text):
    return llm.sentiment(body.text[:1000])


@app.get("/")
def home():  # local dev only; on Vercel, /public is served statically
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "public", "index.html"))
