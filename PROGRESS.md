# ChurnGuard — Progress Log

## How to use this file
- Check off each item as it's completed.
- Every item has an **Understanding Checkpoint** — don't check it off until Leo can answer it in his own words in-chat.
- Add new bugs to **Gotchas** the moment they're found, in G-00N format.
- Add a dated note under a phase whenever a real decision gets made.

---

## Current Phase
**Phase 4 — Serving**

---

## Phase 0 — Setup ✅
- [x] Repo created, `uv` initialized, `ruff` configured
- [x] README skeleton with architecture diagram placeholder

**Understanding Checkpoint:** Why does this project use `uv` + `ruff` instead of plain pip/no linter?
> `uv` generates a lockfile that pins exact package versions — reproducible environments on any machine or CI runner. `ruff` is a linter that catches bugs and style issues before runtime, and enforces code quality in CI.

## Phase 1 — Data + EDA ✅
- [x] Dataset selected and documented (IBM Telco Customer Churn, Kaggle)
- [x] EDA notebook (`notebooks/eda.ipynb`)
- [x] Key findings documented

**Understanding Checkpoint:** Why do we define a strict schema now instead of just working with whatever columns the CSV has?
> The model is trained on specific columns with specific types. In production, data comes from multiple sources that can change independently. A schema is a contract — it fails loudly if data doesn't match expectations, rather than letting bad data corrupt predictions silently.

**EDA Key Findings:**
- 7,043 customers, 21 columns
- 26.5% churn rate — class imbalance needs handling
- TotalCharges has 11 blank rows (new customers, tenure=0)
- Strongest churn signals: contract type (42.7% churn on month-to-month), low tenure, high monthly charges
- TotalCharges is redundant with tenure — may drop in feature engineering

## Phase 2 — Reproducible Training ✅
- [x] DVC initialized, raw dataset tracked (`data/raw/telco_churn.csv`)
- [x] Training script (`training/train.py`) with MLflow logging
- [x] Model registered in MLflow Model Registry as `churnguard-xgboost v1`

**Understanding Checkpoint:** What breaks in a real team if you skip DVC/MLflow and just email around `.pkl` files?
> No reproducibility (can't recreate a run), no audit trail (who trained what on which data), no rollback (which pkl was the good one?), and training/serving skew risk (data processed differently offline vs online).

**Metrics — baseline run:**
- ROC-AUC: 0.843
- Recall: 0.775
- Precision: 0.522
- F1: 0.624
- Accuracy: 0.752

## Phase 3 — Feature Store ✅
- [x] Feast feature definitions written (`feature_repo/feature_definitions.py`)
- [x] Offline store populated (parquet file with 7,043 rows)
- [x] Online store populated (`populate_online_store.py`)

**Understanding Checkpoint:** What is training/serving skew, and how does Feast prevent it?
> Training/serving skew is when a feature is computed differently offline vs online, causing the model to see different data in production than it was trained on. Feast prevents it by writing feature logic once — the same code path is used for both training and serving.

**Decisions:**
- 2026-08-17: Used `store.write_to_online_store()` instead of `feast materialize` due to Feast 0.65 bug with static timestamps (see G-001)

## Phase 4 — Serving
- [ ] FastAPI app returns prediction + SHAP values
- [ ] Dockerfile builds and runs locally

**Understanding Checkpoint:** Why return SHAP values from the API instead of just the probability?

## Phase 5 — LLM Explanation Layer
- [ ] Groq wired in
- [ ] SHAP values → plain-English explanation

**Understanding Checkpoint:** Why feed SHAP values into the prompt instead of just asking the LLM "why might this customer churn?"

## Phase 6 — Agentic Retention Recommender
- [ ] LangGraph graph defined (state, nodes, edges)
- [ ] Tools implemented: crm_lookup, draft_email, escalate_to_am
- [ ] End-to-end run: prediction → explanation → agent decision

**Understanding Checkpoint:** What makes this "agentic" rather than just an LLM call with a system prompt?

## Phase 7 — CI/CD
- [ ] GitHub Actions: lint/test
- [ ] GitHub Actions: train + validate metrics against threshold
- [ ] GitHub Actions: build + push image

**Understanding Checkpoint:** What should happen if the newly trained model's metrics are worse than production's? Why?

## Phase 8 — Kubernetes Deployment
- [ ] Manifests/Helm chart written
- [ ] Deployed to local k3s/minikube

**Understanding Checkpoint:** Why Kubernetes here instead of just running the Docker container on a single VM?

## Phase 9 — Monitoring + Retraining Trigger
- [ ] Evidently drift dashboard wired to prediction logs
- [ ] Langfuse tracing wired to agent/LLM calls
- [ ] Drift threshold → retraining trigger

**Understanding Checkpoint:** What's the difference between what Evidently monitors and what Langfuse monitors?

## Phase 10 — Portfolio Polish
- [ ] Architecture diagram finalized
- [ ] Demo GIF/video recorded
- [ ] README rewritten for recruiter skim-read

---

## Gotchas

**G-001: `feast materialize` writes 0 rows with static parquet data (Feast 0.65)**
When all `event_timestamp` values in the parquet are identical and static, Feast's dask offline store silently skips all rows during materialization.
Workaround: use `store.write_to_online_store()` directly — see `populate_online_store.py`.

---

## Decisions Log

- **2026-08-12**: Chose IBM Telco Customer Churn dataset — realistic features, known class imbalance, industry standard benchmark
- **2026-08-12**: Chose XGBoost over neural net — tabular data, SHAP interpretability required for LLM explanation layer
- **2026-08-15**: Downgraded pandas to 2.2.3 to resolve MLflow compatibility (MLflow 2.19 requires pandas<3)
- **2026-08-15**: Set `UV_LINK_MODE=copy` to fix OneDrive hardlink conflict on Windows
- **2026-08-17**: Used `write_to_online_store()` instead of `feast materialize` — Feast 0.65 bug with static timestamps