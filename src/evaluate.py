from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def main():
    ap = argparse.ArgumentParser(description="Evaluate whether an MLflow run passes promotion thresholds.")
    ap.add_argument("--run-id", required=True, help="MLflow run ID to evaluate")
    ap.add_argument("--min-f1-macro", type=float, default=0.80)
    ap.add_argument("--min-accuracy", type=float, default=0.50)
    ap.add_argument("--output", default="reports/eval_gate.json", help="Where to save evaluation gate result")
    args = ap.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("Missing MLFLOW_TRACKING_URI env var")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    run = client.get_run(args.run_id)
    metrics = run.data.metrics

    test_accuracy = metrics.get("test_accuracy")
    test_f1_macro = metrics.get("test_f1_macro")

    reasons = []
    passed = True

    if test_accuracy is None:
        passed = False
        reasons.append("Missing metric: test_accuracy")
    elif test_accuracy < args.min_accuracy:
        passed = False
        reasons.append(
            f"test_accuracy {test_accuracy:.4f} < threshold {args.min_accuracy:.4f}"
        )

    if test_f1_macro is None:
        passed = False
        reasons.append("Missing metric: test_f1_macro")
    elif test_f1_macro < args.min_f1_macro:
        passed = False
        reasons.append(
            f"test_f1_macro {test_f1_macro:.4f} < threshold {args.min_f1_macro:.4f}"
        )

    result = {
        "run_id": args.run_id,
        "tracking_uri": tracking_uri,
        "metrics": {
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1_macro,
        },
        "thresholds": {
            "test_accuracy": args.min_accuracy,
            "test_f1_macro": args.min_f1_macro,
        },
        "passed": passed,
        "reasons": reasons,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()