"""Synthetic retail data generator (replace with your own CSV / database in production)."""
from functools import lru_cache
import numpy as np
import pandas as pd

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = {"Electronics": (220, .18), "Fashion": (60, .35), "Grocery": (25, .12),
              "Home": (90, .28), "Sports": (70, .25)}


@lru_cache(maxsize=1)
def get_orders() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    t = np.arange(len(days))
    lam = 30 + 0.02 * t + 6 * np.sin(2 * np.pi * t / 365.25) + 5 * (days.dayofweek >= 5)
    lam = lam * np.where(days.month >= 11, 1.3, 1.0)
    counts = rng.poisson(lam)
    day_idx = np.repeat(t, counts)

    n_cust = 900
    weights = rng.gamma(1.2, 1, n_cust)
    start = np.where(rng.random(n_cust) < .5, rng.integers(0, 400, n_cust), 0)
    end = np.where(rng.random(n_cust) < .3, rng.integers(150, 700, n_cust), 10 ** 6)  # churners
    cust = rng.choice(n_cust, size=len(day_idx), p=weights / weights.sum())
    ok = (day_idx >= start[cust]) & (day_idx <= end[cust])
    day_idx, cust = day_idx[ok], cust[ok]
    n = len(day_idx)

    names = list(CATEGORIES)
    cat = rng.choice(len(names), size=n, p=[.15, .25, .30, .15, .15])
    base = np.array([CATEGORIES[c][0] for c in names])[cat]
    margin = np.array([CATEGORIES[c][1] for c in names])[cat]
    price = np.round(base * rng.lognormal(0, .25, n), 2)
    qty = rng.integers(1, 5, n)
    revenue = np.round(price * qty, 2)
    profit = np.round(revenue * (margin + rng.normal(0, .04, n)), 2)

    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "date": days[day_idx],
        "customer_id": cust + 1000,
        "region": rng.choice(REGIONS, size=n, p=[.3, .2, .25, .25]),
        "category": np.array(names)[cat],
        "quantity": qty,
        "unit_price": price,
        "revenue": revenue,
        "profit": profit,
    })
    for d in (150, 320, 560):  # injected anomalies (flash sales / errors)
        m = day_idx == d
        df.loc[m, ["revenue", "profit"]] *= 3
    return df
