import json
from collections import Counter
from pathlib import Path

LOG_PATH = Path("/opt/ticket-router/logs/predictions.jsonl")

def compute_stats(n=200):

    if not LOG_PATH.exists():
        return {"error": "no logs found"}

    lines = LOG_PATH.read_text().splitlines()[-n:]

    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except:
            continue

    if not rows:
        return {"error": "no valid log rows"}

    queues = [r["predicted_queue"] for r in rows]
    confs = [r["confidence"] for r in rows]
    lats  = [r["latency_ms"] for r in rows]

    queue_counts = Counter(queues)
    version_counts = Counter([str(r["model_version"]) for r in rows])

    return {
        "rows_analyzed": len(rows),
        "model_versions": dict(version_counts),
        "top_queues": dict(queue_counts),
        "confidence_mean": sum(confs)/len(confs),
        "latency_mean": sum(lats)/len(lats)
    }