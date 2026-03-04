import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-name", default="ticket-router")
    ap.add_argument("--min-f1-macro", type=float, default=0.80)
    ap.add_argument("--metric-name", default="test_f1_macro")
    ap.add_argument("--artifact-path", default="model", help="Where the model was logged in the run (default: model)")
    args = ap.parse_args()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # 1) Fetch run + metric
    run = client.get_run(args.run_id)
    score = run.data.metrics.get(args.metric_name)

    if score is None:
        print(f"⚠️ Metric '{args.metric_name}' not found on run {args.run_id}. Will still register, but won't stage.")
    else:
        print(f"Metric {args.metric_name}={score:.4f} (threshold={args.min_f1_macro:.4f})")

    # 2) Ensure registered model exists
    try:
        client.get_registered_model(args.model_name)
    except Exception:
        client.create_registered_model(args.model_name)
        print(f"✅ Created registered model: {args.model_name}")

    # 3) Create a model version from the run artifact location
    source = f"{run.info.artifact_uri}/{args.artifact_path}"
    mv = client.create_model_version(
        name=args.model_name,
        source=source,
        run_id=args.run_id,
    )

    print(f"✅ Registered {args.model_name} v{mv.version} from run {args.run_id}")
    print(f"   source: {source}")

    # 4) Stage if metric passes
    if score is not None and score >= args.min_f1_macro:
        client.transition_model_version_stage(
            name=args.model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        print(f"✅ Promoted to Staging: {args.model_name} v{mv.version}")
    else:
        print("ℹ️ Not promoted to Staging.")

if __name__ == "__main__":
    main()