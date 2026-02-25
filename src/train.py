from __future__ import annotations
import os
import time
import yaml
import pandas as pd
import numpy as np

import mlflow
import math
import torch
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from data import DataConfig, load_and_prepare
from mlflow_utils import save_confusion_matrix, save_top_errors


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ds, label2id, id2label = load_and_prepare(
        DataConfig(
            dataset_name=cfg["dataset_name"],
            text_fields=cfg["text_fields"],
            label_field=cfg["label_field"],
            language_field=cfg["language_field"],
            seed=cfg["train"]["seed"],
            test_size=cfg["train"]["test_size"],
            val_size=cfg["train"]["val_size"],
        )
    )

    # --- class weights (to handle imbalance) ---
    train_labels = ds["train"]["label"]
    counts = Counter(train_labels)
    num_labels = len(label2id)

    # weight for class c = total / (num_labels * count[c])
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (num_labels * counts.get(i, 1)) for i in range(num_labels)],
        dtype=torch.float,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg["max_length"])

    ds_tok = ds.map(tok, batched=True)
    keep = {"input_ids", "attention_mask", "label", "text", cfg["language_field"]}
    drop_cols = [c for c in ds_tok["train"].column_names if c not in keep]
    ds_tok = ds_tok.remove_columns(drop_cols)
    ds_tok.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    steps_per_epoch = math.ceil(len(ds["train"]) / cfg["train"]["batch_size"])
    total_steps = steps_per_epoch * int(cfg["train"]["epochs"])
    warmup_steps = int(cfg["train"]["warmup_ratio"] * total_steps)

    args = TrainingArguments(
        output_dir="outputs",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=cfg["train"]["batch_size"],
        per_device_eval_batch_size=cfg["train"]["batch_size"],
        num_train_epochs=cfg["train"]["epochs"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
    )

    class WeightedLossTrainer(Trainer):
        def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            # transformers datasets sometimes use "label"
            if labels is None and "label" in inputs:
                labels = inputs["label"]
                inputs = {k: v for k, v in inputs.items() if k != "label"}
                inputs["labels"] = labels

            outputs = model(**inputs)
            logits = outputs.get("logits")

            weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(logits, labels)

            return (loss, outputs) if return_outputs else loss

    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=class_weights,
    )

    # MLflow uses env vars you already set on tr-train-01:
    # MLFLOW_TRACKING_URI, MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    exp_name = cfg["mlflow"]["experiment"]
    mlflow.set_experiment(exp_name)

    run_name = f'{cfg["project_name"]}-{cfg["model_name"]}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_name": cfg["model_name"],
            "max_length": cfg["max_length"],
            "num_labels": len(label2id),
            **{f"train_{k}": v for k, v in cfg["train"].items()},
        })

        os.makedirs("reports", exist_ok=True)

        pd.DataFrame(
            {"label_id": list(range(num_labels)),
            "queue": [id2label[i] for i in range(num_labels)],
            "count": [counts.get(i, 0) for i in range(num_labels)],
            "weight": class_weights.tolist()}
            ).to_csv("reports/class_weights.csv", index=False)
        
        mlflow.log_artifact("reports/class_weights.csv")
        
        t0 = time.time()
        trainer.train()
        mlflow.log_metric("train_time_sec", time.time() - t0)

        # Test metrics + artifacts
        pred = trainer.predict(ds_tok["test"])
        y_true = pred.label_ids
        y_pred = np.argmax(pred.predictions, axis=1)

        test_metrics = compute_metrics((pred.predictions, pred.label_ids))
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", float(v))

        cm_path = "reports/confusion_matrix.png"
        err_path = "reports/errors.csv"

        save_confusion_matrix(y_true, y_pred, id2label, cm_path)
        save_top_errors(ds["test"], y_true, y_pred, id2label, err_path, n=50)

        mlflow.log_artifact("configs/base.yaml")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(err_path)

        trainer.save_model("outputs/best_model")
        mlflow.log_artifacts("outputs/best_model", artifact_path="model")

    print("Done. Check MLflow UI for run + artifacts.")


if __name__ == "__main__":
    main()