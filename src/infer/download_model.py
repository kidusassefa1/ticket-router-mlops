from __future__ import annotations

import os, json, time, argparse
import mlflow
from mlflow.tracking import MlflowClient


def main():
    ap = argparse.ArgumentParser(description="Download MLflow model by name+version OR name+stage.")
    ap.add_argument("--model-name", default="ticket-router")
    ap.add_argument("--version", help="Model version number (ex: 3).")
    ap.add_argument("--stage", help="Model stage (ex: Staging or Production).")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    if (args.version is None) == (args.stage is None):
        raise SystemExit("Provide exactly one: --version OR --stage")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("Missing MLFLOW_TRACKING_URI env var.")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if args.stage:
        latest = client.get_latest_versions(args.model_name, stages=[args.stage])
        if not latest:
            raise SystemExit(f"No versions found for {args.model_name} in stage {args.stage}")
        mv = latest[0]
        model_uri = f"models:/{args.model_name}/{args.stage}"
        resolved_version = str(mv.version)
        run_id = mv.run_id
    else:
        mv = client.get_model_version(args.model_name, args.version)
        model_uri = f"models:/{args.model_name}/{args.version}"
        resolved_version = str(mv.version)
        run_id = mv.run_id

    os.makedirs(args.dst, exist_ok=True)

    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=args.dst,
    )

    meta = {
        "model_name": args.model_name,
        "requested_stage": args.stage,
        "requested_version": args.version,
        "resolved_version": resolved_version,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "downloaded_at_unix": int(time.time()),
        "download_path": local_path,
        "model_uri": model_uri,
    }

    meta_path = os.path.join(args.dst, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Downloaded: {model_uri}")
    print(f"✅ Saved to:   {local_path}")
    print(f"✅ Metadata:   {meta_path}")


if __name__ == "__main__":
    main()