# Milestones

## Completed

### Milestone 0 — Homelab + GPU foundation
- Proxmox homelab running
- GPU passthrough to training VM verified (nvidia-smi shows utilization)

### Milestone 1 — Project repo structure
- Production-style repo layout (src/, configs/, infra/, docs/, reports/)

### Milestone 2 — MLOps stack (MLflow + Postgres + MinIO)
- Docker Compose running on tr-mlops-01
- Training VM logs experiments to MLflow
- Artifacts stored in MinIO via S3 API

### Milestone 3 — Baseline model training
- XLM-R baseline fine-tuned
- Logged metrics to MLflow

### Milestone 4 — Model improvement
- Added class-weighted loss
- Trained 3 epochs
- Improved macro F1 significantly
- Saved confusion matrix + error examples

### Milestone 5 — Production inference deployment
- Inference service runs on tr-infer-01
- Downloads a model from MLflow artifacts
- Exposes /health and /predict endpoints

## Next (Enterprise track)

### Milestone 6 — Model versioning + promotion
- MLflow Model Registry
- “Staging” → “Production” promotion
- Inference pulls the “Production” model automatically

### Milestone 7 — Observability + reliability
- Structured logging
- Metrics endpoint (Prometheus)
- Latency + error rate dashboards
- Basic load testing

### Milestone 8 — CI/CD
- GitHub Actions for lint/tests
- Build/push inference container
- Deployment automation

### Milestone 9 — Responsible AI
- Dataset/label risks documented
- Bias checks by language/category
- Model card finalized + limitations