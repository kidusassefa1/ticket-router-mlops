# MLOps Stack (MLflow + Postgres + MinIO)

## Where it runs
- mlops-01 (VM)
- Docker Compose in: `infra/mlops-services/`

## What each service does
- Postgres: MLflow backend store (run metadata)
- MinIO: S3-compatible artifact store (models, plots, reports)
- MLflow: tracking server + UI

## How artifacts get stored
Training code uses MLflow to log artifacts.
MLflow uses S3 credentials + endpoint to upload files to MinIO.

Key env vars:
- MLFLOW_TRACKING_URI
- MLFLOW_S3_ENDPOINT_URL
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
- AWS_S3_ADDRESSING_STYLE=path