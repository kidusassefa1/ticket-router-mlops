# Inference Service

## Where it runs
- tr-infer-01

## What it does
- Downloads a selected MLflow run model artifact
- Loads tokenizer + model
- Serves predictions via FastAPI

Endpoints:
- GET /health
- POST /predict

## Example request
```json
{
  "subject": "Cannot login",
  "body": "Password reset fails and MFA does not work."
}
```

## Example response

- predicted queue
- confidence
- top-k alternatives
- latency in ms