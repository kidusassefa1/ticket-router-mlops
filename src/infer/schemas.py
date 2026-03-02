from pydantic import BaseModel
from typing import Optional, List, Dict

class PredictRequest(BaseModel):
    subject: Optional[str] = ""
    body: Optional[str] = ""
    language: Optional[str] = None  # optional metadata

class PredictResponse(BaseModel):
    predicted_queue: str
    confidence: float
    top_k: List[Dict[str, float]]
    latency_ms: float