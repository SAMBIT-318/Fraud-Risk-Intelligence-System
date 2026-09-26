# 🛡️ Fraud Risk Intelligence System

> Hybrid ML + GenAI fraud detection platform — XGBoost classifier with SHAP explainability and Anthropic Claude AI investigation reports.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.45-green)

## Features

| Feature | Detail |
|---------|--------|
| **Fraud detection** | XGBoost + SMOTE, trained on 50,000 transactions (1:578 imbalance handled) |
| **Explainability** | SHAP TreeExplainer — waterfall plot + top-5 risk factors per transaction |
| **AI reports** | Gemini (optional) writes investigation reports grounded in SHAP values |
| **Bulk scoring** | Upload any CSV, score all rows, download results with risk scores |
| **Performance dashboard** | ROC curve, PR curve, confusion matrix, feature importance |

## Quick start — Streamlit Cloud (free, recommended)

1. **Fork this repo** on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your fork, branch `main`, file `app.py`
4. In **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "sk-ant-your-key-here"
   ```
5. Click **Deploy** — live in ~2 minutes

## Local setup

```bash
git clone https://github.com/SAMBIT-318/fraud-risk-intelligence.git
cd fraud-risk-intelligence

pip install -r requirements.txt

# Optional: add Gemini API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and replace the placeholder key

streamlit run app.py
```

## Docker

```bash
# With Gemini (set your key in .env or export GEMINI_API_KEY=sk-ant-...)
docker-compose up --build

# App available at http://localhost:8501
```

## Using the real Kaggle dataset (optional)

The app generates 50,000 synthetic transactions by default.
For the full 284,807-transaction dataset:

1. Download `creditcard.csv` from [Kaggle ULB Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it in the `data/` folder
3. Restart the app — it auto-detects the real dataset

## Architecture

```
Raw transaction (Amount, Time, V1–V28)
         ↓
Feature engineering  →  Amount_log, Amount_zscore, Hour, Is_night  (32 features)
         ↓
StandardScaler  →  SMOTE (train only)  →  XGBoost (200 estimators)
         ↓
Risk score (0–100)  +  SHAP TreeExplainer  →  top-5 feature attributions
         ↓
Gemini Google-genai gemini 2.5 flash  →  structured investigation report
         ↓
Streamlit (3 tabs): Analyzer | Bulk Scoring | Performance Dashboard
```

## Risk thresholds

| Score | Verdict |
|-------|---------|
| 0–30 | Legitimate |
| 31–60 | Needs review |
| 61–85 | High risk |
| 86–100 | Confirmed fraud |

## Skills demonstrated

`Python` · `scikit-learn` · `XGBoost` · `SMOTE / imbalanced-learn` · `SHAP` · `Gemini API` · `Streamlit` · `Docker` · `Feature Engineering` · `Predictive Modeling` · `Plotly` · `Pandas` · `NumPy`

## Author

**Sambit Swain** — MCA Candidate | Data Science & AI/GenAI

[GitHub](https://github.com/SAMBIT-318) · [LinkedIn](https://www.linkedin.com/in/sambit-swain-1a6151358)
