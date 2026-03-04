import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="ticket-router")
    ap.add_argument("--version", required=True)
    ap.add_argument("--archive-existing", action="store_true")
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    client.transition_model_version_stage(
        name=args.model_name,
        version=args.version,
        stage="Production",
        archive_existing_versions=args.archive_existing,
    )
    print(f"✅ Promoted to Production: {args.model_name} v{args.version}")

if __name__ == "__main__":
    main()