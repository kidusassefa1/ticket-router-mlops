# Ticket Router MLOps

A production-style AI/ML system that trains a transformer model to route customer support tickets into the correct **queue** (52 classes), tracks experiments in **MLflow**, stores artifacts in **MinIO**, and serves predictions via a **FastAPI inference service**.

This project is built to mimic a real enterprise AI/ML engineering workflow using a Proxmox homelab.

## What this system does

- Fine-tunes a transformer model (XLM-R) for multi-class text classification (ticket → queue)
- Logs metrics, parameters, and artifacts to MLflow
- Stores artifacts (model files, plots, reports) in MinIO (S3-compatible)
- Stores MLflow metadata in Postgres
- Deploys an inference API that downloads a chosen model from MLflow artifacts and serves:
  - `GET /health`
  - `POST /predict`

---

## Architecture (high level)

**VMs / Services**
- **mlops-01 (VM)**: Docker Compose stack running:
  - MLflow (tracking UI)
  - Postgres (MLflow metadata)
  - MinIO (artifact storage)
- **train-01 (VM, GPU passthrough)**:
  - model training (PyTorch + Hugging Face)
  - logs to MLflow + uploads artifacts to MinIO
- **infer-01 (LXC/VM)**:
  - FastAPI inference service
  - downloads model artifacts from MLflow

See `docs/01_architecture.md` for details.

---

## Results (current best)

- **num_labels (queues):** 52
- **baseline:** test_accuracy ~0.49, test_f1_macro ~0.68
- **improved (class-weighted + 3 epochs):**
  - test_accuracy: **0.509**
  - test_f1_macro: **0.832**
  - test_f1_weighted: **0.509**

Artifacts include confusion matrix and error samples.

See `reports/eval_report.md`.

---

## How to run (quick start)

### 1) MLOps stack (on tr-mlops-01)
From `infra/mlops-services/`:

```bash
docker compose up -d
```
MLflow: http://10.0.0.73:5000

MinIO Console: http://10.0.0.73:9001

### 2) Training (on tr-train-01)
From `infra/mlops-services/`:

```bash
cd ~/ticket-router-mlops
source .venv/bin/activate

set -a
source .env
set +a

python src/train.py
```

### 3) Inference API (on tr-infer-01)
Download a model from MLflow by run_id, then run FastAPI:

```bash
set -a
source .env
set +a

python -m src.infer.download_model --run-id <RUN_ID> --dst /opt/ticket-router/model
uvicorn src.infer.app:app --host 0.0.0.0 --port 8000
```
Test:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject":"Cannot login", "body":"MFA fails and password reset does not work."}'
```

## Repo Guide

- `src/` — training + evaluation + inference code
- `configs/` — training configs
- `infra/` — docker compose stack for MLflow/Postgres/MinIO
- `docs/` — architecture, milestones, runbooks
- `reports/` — model results and analysis outputs

## Notes

- This project is designed to show end-to-end ML engineering: `training → tracking → artifacts → deployment`.
- Next steps (enterprise-ready):
    - MLflow Model Registry + “Production” stage promotion
    - Containerize inference + systemd
    - Monitoring/metrics (latency, error rates, drift checks)
    - CI/CD with GitHub Actions
    
See `docs/00_milestones.md`.
