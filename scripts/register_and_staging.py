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
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    run = client.get_run(args.run_id)
    metrics = run.data.metrics
    score = metrics.get(args.metric_name)

    mv = mlflow.register_model(f"runs:/{args.run_id}/model", args.model_name)
    print(f"✅ Registered {args.model_name} v{mv.version} from run {args.run_id}")

    if score is None:
        print(f"⚠️ Metric {args.metric_name} not found on run. Not staging.")
        return

    print(f"Metric {args.metric_name}={score:.4f} (threshold={args.min_f1_macro:.4f})")
    if score >= args.min_f1_macro:
        client.transition_model_version_stage(
            name=args.model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        print(f"✅ Promoted to Staging: {args.model_name} v{mv.version}")
    else:
        print("❌ Did not meet threshold. Left un-staged.")

if __name__ == "__main__":
    main()