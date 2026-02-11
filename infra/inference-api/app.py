import os
import mlflow
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = os.getenv("MODEL_NAME", "ticket-router-xlmrb")  # MLflow registered model name
STAGE = os.getenv("MODEL_STAGE", "Production")

app = FastAPI(title="Ticket Router API")

tokenizer = None
model = None

class PredictIn(BaseModel):
    text: str
    top_k: int = 3

def load_production_model():
    global tokenizer, model

    client = mlflow.tracking.MlflowClient()
    mv = client.get_latest_versions(MODEL_NAME, stages=[STAGE])
    if not mv:
        raise RuntimeError(f"No model version found for {MODEL_NAME} in stage {STAGE}")

    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_path)
    model.eval()

@app.on_event("startup")
def startup_event():
    load_production_model()

@app.post("/predict")
def predict(inp: PredictIn):
    assert tokenizer is not None and model is not None

    inputs = tokenizer(inp.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0)

    top_k = min(inp.top_k, probs.shape[0])
    vals, idxs = torch.topk(probs, k=top_k)

    results = []
    for score, idx in zip(vals.tolist(), idxs.tolist()):
        label = model.config.id2label.get(idx, str(idx)) if isinstance(model.config.id2label, dict) else model.config.id2label[idx]
        results.append({"queue": label, "prob": float(score)})

    return {"predicted": results[0], "top_k": results}