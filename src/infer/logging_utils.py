from pathlib import Path
import json
import time

LOG_PATH = Path("/opt/ticket-router/logs/predictions.jsonl")

def log_prediction(record: dict):
    record["timestamp"] = time.time()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")