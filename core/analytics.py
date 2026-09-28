"""Lightweight analytics: KPIs, forecasting, anomalies, RFM segmentation, churn model."""
import numpy as np
import pandas as pd


# ---------- KPIs ----------
def kpis(df):
    s = df.groupby("date")["revenue"].sum()
    last30, prev30 = s.iloc[-30:].sum(), s.iloc[-60:-30].sum()
    monthly = df.groupby(df["date"].dt.strftime("%Y-%m"))["revenue"].sum().round(0)
    return {
        "revenue": round(float(df.revenue.sum()), 2), "profit": round(float(df.profit.sum()), 2),
        "orders": int(len(df)), "customers": int(df.customer_id.nunique()),
        "aov": round(float(df.revenue.mean()), 2),
        "growth_30d_pct": round(float((last30 / prev30 - 1) * 100), 2) if prev30 else 0,
        "by_region": df.groupby("region")["revenue"].sum().round(0).to_dict(),
        "by_category": df.groupby("category")["revenue"].sum().round(0).to_dict(),
        "monthly": {"labels": list(monthly.index), "values": monthly.tolist()},
    }


# ---------- Forecasting ----------
def daily_series(df):
    return df.groupby("date")["revenue"].sum().asfreq("D", fill_value=0.0)


def _design(idx, t0):
    t = np.asarray((idx - t0).days, dtype=float)
    w = 2 * np.pi * t / 365.25
    return np.column_stack([t / 365.25, np.eye(7)[np.asarray(idx.dayofweek)], np.sin(w), np.cos(w)])


def _reg_predict(train, future_idx):
    y = train.clip(upper=train.quantile(0.99)).values  # winsorize spikes
    X = _design(train.index, train.index[0])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = pd.Series(_design(future_idx, train.index[0]) @ coef, index=future_idx)
    return pred, X @ coef


def _naive(train, idx):
    return pd.Series(train.tail(28).mean(), index=idx)


def _snaive(train, idx):
    last = train.tail(7).values
    return pd.Series(last[np.arange(len(idx)) % 7], index=idx)


MODELS = {
    "Moving Average (28d)": _naive,
    "Seasonal Naive (weekly)": _snaive,
    "Trend + Seasonality Regression": lambda tr, idx: _reg_predict(tr, idx)[0],
}


def forecast(df, horizon=30):
    horizon = max(7, min(int(horizon), 90))
    s = daily_series(df)
    train, test = s.iloc[:-30], s.iloc[-30:]
    board = []
    for name, fn in MODELS.items():
        p = fn(train, test.index)
        rmse = float(np.sqrt(np.mean((test - p) ** 2)))
        nz = test > 0
        mape = float(np.mean(np.abs((test[nz] - p[nz]) / test[nz])) * 100)
        board.append({"model": name, "rmse": round(rmse, 1), "mape": round(mape, 2)})
    best = min(board, key=lambda r: r["rmse"])
    fidx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon)
    pred = MODELS[best["model"]](s, fidx)
    band = 1.96 * best["rmse"]
    hist = s.iloc[-90:]
    return {
        "best_model": best["model"], "leaderboard": sorted(board, key=lambda r: r["rmse"]),
        "history": {"dates": [d.strftime("%Y-%m-%d") for d in hist.index], "values": hist.round(0).tolist()},
        "forecast": {"dates": [d.strftime("%Y-%m-%d") for d in fidx],
                     "values": pred.round(0).tolist(),
                     "lower": (pred - band).clip(lower=0).round(0).tolist(),
                     "upper": (pred + band).round(0).tolist()},
    }


# ---------- Anomaly detection ----------
def anomalies(df, threshold=4.0):
    s = daily_series(df)
    _, fit = _reg_predict(s, s.index[-1:] + pd.Timedelta(days=1))
    resid = s.values - fit
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) or 1.0
    z = 0.6745 * (resid - med) / mad
    out = pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "revenue": s.values.round(0),
                        "expected": fit.round(0), "score": z.round(1)})
    out = out[out.score.abs() > threshold]
    out = out.reindex(out.score.abs().sort_values(ascending=False).index)
    return out.head(15).to_dict("records")


# ---------- RFM segmentation ----------
def rfm(df):
    snap = df.date.max() + pd.Timedelta(days=1)
    g = df.groupby("customer_id").agg(last=("date", "max"), frequency=("order_id", "count"),
                                      monetary=("revenue", "sum"))
    g["recency"] = (snap - g["last"]).dt.days

    def q(s):
        return pd.qcut(s.rank(method="first"), 5, labels=False) + 1

    g["R"], g["F"], g["M"] = 6 - q(g.recency), q(g.frequency), q(g.monetary)
    g["segment"] = np.select(
        [(g.R >= 4) & (g.F >= 4), (g.R >= 3) & (g.F >= 3), (g.R >= 4) & (g.F <= 2),
         (g.R <= 2) & (g.F >= 4), g.R <= 2],
        ["Champions", "Loyal", "New / Promising", "At Risk (high value)", "Hibernating"],
        default="Needs Attention")
    summary = g.groupby("segment").agg(customers=("R", "size"), avg_revenue=("monetary", "mean"),
                                       avg_recency_days=("recency", "mean")).round(1)
    return summary.reset_index().sort_values("customers", ascending=False).to_dict("records")


# ---------- Churn model (logistic regression from scratch) ----------
def _features(d, ref):
    g = d.groupby("customer_id").agg(last=("date", "max"), first=("date", "min"),
                                     frequency=("order_id", "count"), monetary=("revenue", "sum"))
    g["recency"] = (ref - g["last"]).dt.days
    g["tenure"] = (ref - g["first"]).dt.days
    g["avg_order"] = g.monetary / g.frequency
    return g[["recency", "frequency", "monetary", "avg_order", "tenure"]]


def _auc(y, p):
    r = pd.Series(p).rank().values
    pos, neg = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else 0.5


def churn(df, window=90):
    cutoff = df.date.max() - pd.Timedelta(days=window)
    X = _features(df[df.date <= cutoff], cutoff)
    y = (~X.index.isin(df[df.date > cutoff].customer_id.unique())).astype(int)
    L = np.log1p(X.values)
    mu, sd = L.mean(0), L.std(0) + 1e-9
    Z = (L - mu) / sd
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Z))
    tr, te = idx[: int(.75 * len(Z))], idx[int(.75 * len(Z)):]
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(1500):
        p = 1 / (1 + np.exp(-(Z[tr] @ w + b)))
        w -= 0.1 * (Z[tr].T @ (p - y[tr]) / len(tr))
        b -= 0.1 * np.mean(p - y[tr])
    pt = 1 / (1 + np.exp(-(Z[te] @ w + b)))
    now = _features(df, df.date.max())
    Zn = (np.log1p(now.values) - mu) / sd
    now["churn_risk"] = (1 / (1 + np.exp(-(Zn @ w + b)))).round(3)
    risky = now[now.recency <= 120].sort_values("churn_risk", ascending=False).head(15)
    return {
        "auc": round(_auc(y[te], pt), 3), "accuracy": round(float(np.mean((pt > .5) == y[te])), 3),
        "churn_rate": round(float(y.mean()), 3),
        "drivers": sorted([{"feature": f, "weight": round(float(v), 3)} for f, v in zip(X.columns, w)],
                          key=lambda d: -abs(d["weight"])),
        "at_risk": risky.reset_index().round(1).to_dict("records"),
    }


# ---------- NLP sentiment (lexicon fallback; see llm.py for Hugging Face) ----------
POS = {"good", "great", "excellent", "love", "amazing", "fast", "best", "happy", "awesome", "perfect", "nice"}
NEG = {"bad", "poor", "terrible", "hate", "slow", "worst", "broken", "awful", "late", "refund", "disappointed"}


def lexicon_sentiment(text):
    words = [w.strip(".,!?").lower() for w in text.split()]
    score = sum(w in POS for w in words) - sum(w in NEG for w in words)
    return {"label": "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL",
            "score": score, "engine": "lexicon"}
