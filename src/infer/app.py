from __future__ import annotations
import os, time, json
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .schemas import PredictRequest, PredictResponse
from .utils import build_text, predict_topk

APP_VERSION = "0.1.0"
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ticket-router/model/model")

app = FastAPI(title="Ticket Router Inference API", version=APP_VERSION)

_model = None
_tok = None
_id2label = None

def load_model():
    global _model, _tok, _id2label
    if _model is not None:
        return

    _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device)
    _model.eval()

    # Prefer config id2label if present
    cfg = _model.config
    _id2label = getattr(cfg, "id2label", None)
    if _id2label is None:
        _id2label = {i: str(i) for i in range(cfg.num_labels)}

@app.get("/health")
def health():
    load_model()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "device": str(next(_model.parameters()).device),
        "num_labels": int(_model.config.num_labels),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    load_model()
    t0 = time.time()

    text = build_text(req.subject, req.body)
    pred_id, conf, top = predict_topk(_model, _tok, text, top_k=5)

    predicted_queue = _id2label.get(pred_id, str(pred_id))
    top_k = [{_id2label.get(i, str(i)): float(p)} for i, p in top]

    latency_ms = (time.time() - t0) * 1000.0
    return PredictResponse(
        predicted_queue=predicted_queue,
        confidence=float(conf),
        top_k=top_k,
        latency_ms=float(latency_ms),
    )