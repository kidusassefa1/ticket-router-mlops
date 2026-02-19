from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_confusion_matrix(y_true, y_pred, id2label: dict[int, str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    labels = sorted(id2label.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(cm, display_labels=[id2label[i] for i in labels])
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_top_errors(ds_test, y_true, y_pred, id2label: dict[int, str], out_path: str, n: int = 50) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if int(t) != int(p):
            ex = ds_test[i]
            rows.append(
                {
                    "index": i,
                    "true_queue": id2label[int(t)],
                    "pred_queue": id2label[int(p)],
                    "language": ex.get("language", None),
                    "text": (ex.get("text", "") or "")[:3000],
                }
            )
        if len(rows) >= n:
            break

    pd.DataFrame(rows).to_csv(out_path, index=False)