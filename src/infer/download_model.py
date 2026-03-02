from __future__ import annotations
import os
import argparse
import mlflow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dst", default="/opt/ticket-router/model")
    args = ap.parse_args()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    os.makedirs(args.dst, exist_ok=True)

    # We logged artifacts under artifact_path="model"
    # This downloads the whole "model/" folder from the run.
    local_path = mlflow.artifacts.download_artifacts(
        run_id=args.run_id,
        artifact_path="model",
        dst_path=args.dst,
    )
    print(f"Downloaded model artifacts to: {local_path}")

if __name__ == "__main__":
    main()