# ChurnGuard 🛡️

> End-to-end MLOps platform for customer churn prediction — featuring a feature store, LLM explanation layer, and agentic retention recommender.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/Serving-FastAPI-green)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-purple)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)

---

## What This Is

ChurnGuard predicts which customers are at risk of churning, explains *why* in plain English, and recommends a concrete retention action — the way a real customer-success platform (Gainsight, ChurnZero) would.

**This is not a notebook project.** It is built as a production ML system: versioned data, a feature store, a model registry, a served API, an LLM agent, drift monitoring, and CI/CD.

---

## Architecture

```
Raw customer/usage events
        │
        ▼
  Feature Engineering → Feast (offline + online store)
        │
        ▼
  Training Job (XGBoost + MLflow)
        │
        ▼
  FastAPI Serving Layer
        │
        ├──► Churn probability + SHAP values
        │
        ▼
  LLM Explanation Agent (Groq)
        │
        ▼
  Agentic Retention Recommender (LangGraph)
        │  tools: crm_lookup, draft_email, escalate_to_am
        ▼
  Monitoring: Evidently (drift) + Langfuse (LLM observability)
        │
        ▼
  Drift threshold → retraining trigger (GitHub Actions)
```

*(Full diagram coming in Phase 10)*

---

## Tech Stack & Why

| Layer | Tool | Why |
|---|---|---|
| Feature store | Feast | Prevents training/serving skew — the #1 silent bug in ML systems |
| Experiment tracking | MLflow | Reproducible runs + model registry, not `model_final_v3.pkl` |
| Model | XGBoost | Best for tabular data; interpretable via SHAP |
| Serving | FastAPI | Async, typed, production-standard |
| LLM explanation | Groq | Turns SHAP values into plain English a CS rep can act on |
| Agent | LangGraph | Explicit state machine — auditable, not an opaque prompt |
| Drift monitoring | Evidently | Detects when incoming data no longer resembles training data |
| LLM observability | Langfuse | Traces agent calls, cost, latency |
| CI/CD | GitHub Actions | Lint, test, train, validate, build, push |

---

## Project Structure

```
churnguard/
├── data/                  # raw + processed (DVC-tracked)
├── feature_repo/          # Feast feature definitions
├── training/              # training scripts, MLflow logging
├── serving/               # FastAPI app
│   ├── main.py
│   ├── model_loader.py
│   └── explain.py         # SHAP + LLM explanation
├── agent/                 # LangGraph retention agent
│   ├── graph.py
│   └── tools.py
├── monitoring/            # Evidently + Langfuse config
├── .github/workflows/     # CI/CD pipelines
├── docker/
├── ARCHITECTURE.md
└── PROGRESS.md
```

---

## How to Run Locally

```powershell
# 1. Clone the repo
git clone https://github.com/codewithleo1/ChurnGaurd.git
cd ChurnGaurd

# 2. Install dependencies
uv sync

# 3. Run the API
uv run uvicorn serving.main:app --reload
```

*(Full setup instructions will be added as each phase completes)*

---

## Key Decisions

- **XGBoost over a neural net** — tabular data doesn't benefit from deep learning; SHAP interpretability is a hard requirement for the LLM explanation layer
- **Feast over querying Postgres twice** — guarantees identical feature logic at training time and serving time
- **LangGraph over a single LLM prompt** — explicit graph state makes agent steps auditable and testable

---

## Gotchas

*(logged as G-001, G-002... as discovered during the build)*

---

## Build Progress

See [PROGRESS.md](./PROGRESS.md) for the full phase-by-phase checklist.

---

## Author

Leo | [GitHub](https://github.com/codewithleo1)