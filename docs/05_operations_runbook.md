# Operations Runbook

## Check MLflow is up
- http://10.0.0.73:5000
- docker compose ps on tr-mlops-01

## Check MinIO is up
- http://10.0.0.73:9001

## Check training GPU
- nvidia-smi on tr-train-01

## Check inference API
- curl http://<infer-ip>:8000/health

## Common issues
- SignatureDoesNotMatch (MinIO):
  - wrong AWS creds in training/inference .env
  - missing MLFLOW_S3_ENDPOINT_URL
  - missing AWS_DEFAULT_REGION or AWS_S3_ADDRESSING_STYLE=path