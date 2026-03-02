from __future__ import annotations
from typing import Tuple, List
import torch

def build_text(subject: str | None, body: str | None) -> str:
    subject = (subject or "").strip()
    body = (body or "").strip()
    if subject and body:
        return f"Subject: {subject}\n\nBody: {body}"
    return subject or body

@torch.inference_mode()
def predict_topk(model, tokenizer, text: str, top_k: int = 5) -> Tuple[int, float, List[Tuple[int, float]]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)

    top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]))
    top = list(zip(top_ids.tolist(), top_probs.tolist()))

    pred_id = int(top[0][0])
    conf = float(top[0][1])
    return pred_id, conf, top