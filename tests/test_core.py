import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from fastapi.testclient import TestClient
from api.index import app
from core import analytics as A, llm
from core.data import get_orders

client = TestClient(app)


def test_data():
    df = get_orders()
    assert len(df) > 10000 and df.revenue.min() > 0


def test_forecast_and_metrics():
    f = A.forecast(get_orders(), 14)
    assert len(f["forecast"]["values"]) == 14 and len(f["leaderboard"]) == 3


def test_anomalies_find_injected_spikes():
    assert len(A.anomalies(get_orders())) >= 3


def test_churn_beats_random():
    assert A.churn(get_orders())["auc"] > 0.6


def test_sql_guard():
    for bad in ["DROP TABLE orders", "SELECT 1; DELETE FROM orders", "UPDATE orders SET revenue=0"]:
        with pytest.raises(ValueError):
            llm.validate_sql(bad)


def test_endpoints():
    for path in ["/api/health", "/api/kpis", "/api/forecast", "/api/segments", "/api/anomalies", "/api/churn"]:
        assert client.get(path).status_code == 200
    r = client.post("/api/ask", json={"question": "Which region had the lowest profit?"})
    assert r.status_code == 200 and r.json()["rows"]
    assert client.post("/api/sentiment", json={"text": "great product"}).json()["label"] == "POSITIVE"
