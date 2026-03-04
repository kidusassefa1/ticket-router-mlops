from __future__ import annotations
import os, time, json
from pathlib import Path
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

def _read_model_meta(model_dir: str) -> dict | None:
    meta_path = Path(model_dir) / "model_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {"error": "failed_to_read_model_meta.json"}

@app.get("/health")
def health():
    load_model()

    meta = _read_model_meta(MODEL_DIR)

    # device (safe even if on CPU)
    device = "unknown"
    try:
        device = str(next(_model.parameters()).device)
    except Exception:
        pass

    return {
        "status": "ok",
        "app_version": APP_VERSION,          # your API code version
        "model_dir": MODEL_DIR,              # should be /opt/ticket-router/current
        "device": device,
        "num_labels": int(getattr(_model.config, "num_labels", -1)),
        "model": meta,                       # name, resolved_version, run_id, etc.
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