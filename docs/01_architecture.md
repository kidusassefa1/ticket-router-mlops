# Architecture

## Goal
Route customer support tickets into the correct queue using a transformer classifier and deploy it like a real internal service.

## Components

### mlops-01 (VM) — “Platform services”
Runs Docker Compose services:

- **MLflow**: tracking UI + tracking server
- **Postgres**: stores experiment metadata (runs, params, metrics)
- **MinIO**: stores artifacts (models, reports, plots) using an S3-compatible API

### train-01 (VM + GPU) — “Training”
- Downloads dataset + model
- Fine-tunes transformer on GPU
- Logs metrics/params to MLflow
- Uploads artifacts to MinIO

### infer-01 (LXC/VM) — “Serving”
- Pulls a chosen model from MLflow artifacts (by run_id)
- Loads tokenizer + model
- Exposes a FastAPI service:
  - GET /health
  - POST /predict

## Data flow (simple)
1. Train VM starts a run in MLflow
2. MLflow stores run metadata in Postgres
3. Train VM uploads model + artifacts to MinIO
4. Inference VM downloads a model artifact and serves predictions