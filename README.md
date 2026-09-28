# AI Decision Intelligence Platform

End-to-end AI data analytics platform: forecasting, anomaly detection, RFM customer segmentation,
churn prediction, sentiment analysis and an LLM "ask your data in English" (text-to-SQL) interface.

**Stack:** Python, FastAPI, Pandas, NumPy, SQLite, Chart.js, Gemini/Groq/Hugging Face APIs, Vercel, Docker, GitHub Actions.

## Features
| Module | Method |
|---|---|
| Forecasting | 3 models compared by RMSE/MAPE on a 30-day backtest (moving average, seasonal naive, trend+seasonality regression) |
| Anomaly detection | Robust MAD z-score on regression residuals |
| Customer segmentation | RFM scoring, 6 segments |
| Churn prediction | Logistic regression written from scratch, AUC reported, weights shown for explainability |
| Ask your data | Gemini or Groq converts English to SQL; SQL is validated (SELECT only); offline fallback if no key |
| NLP | Hugging Face sentiment model, lexicon fallback |

## Run on your PC
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env        # Mac/Linux: cp .env.example .env  (then paste your free keys)
pytest -q
uvicorn api.index:app --reload
```
Open http://127.0.0.1:8000 (API docs at /docs).

## Free API keys (get your own)
- Gemini: https://aistudio.google.com/apikey  -> `GEMINI_API_KEY`
- Groq: https://console.groq.com/keys  -> `GROQ_API_KEY`
- Hugging Face: https://huggingface.co/settings/tokens  -> `HF_TOKEN`

The app works without keys (rule-based fallback). Model names and free limits change, so
override them with `GEMINI_MODEL` / `GROQ_MODEL` in `.env` if needed.

## Push to GitHub
Create an empty repo on github.com first (no README), then:
```bash
git init
git add .
git commit -m "Initial commit: AI Decision Intelligence Platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-decision-platform.git
git push -u origin main
```

## Deploy on Vercel (free)
**Option A - website:** vercel.com -> Add New Project -> import your GitHub repo -> add
environment variables (GEMINI_API_KEY etc.) -> Deploy.

**Option B - command line:**
```bash
npm i -g vercel
vercel login
vercel            # first deploy (preview)
vercel env add GEMINI_API_KEY
vercel --prod
```

## Docker
```bash
docker build -t ai-platform .
docker run -p 8000:8000 --env-file .env ai-platform
```

## Use your own data
Replace `get_orders()` in `core/data.py` with `pd.read_csv("your.csv")`. Keep the columns:
order_id, date, customer_id, region, category, quantity, unit_price, revenue, profit.

## Project structure
```
api/index.py       FastAPI app (Vercel entry point)
core/data.py       data generator / loader
core/analytics.py  forecasting, anomalies, RFM, churn
core/llm.py        text-to-SQL and sentiment
public/index.html  dashboard
tests/             pytest suite
vercel.json  Dockerfile  .github/workflows/ci.yml
```
