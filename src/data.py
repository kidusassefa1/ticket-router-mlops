from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from datasets import load_dataset, DatasetDict


@dataclass
class DataConfig:
    dataset_name: str
    text_fields: list[str]
    label_field: str
    language_field: str
    seed: int
    test_size: float
    val_size: float


def _build_text(ex: Dict[str, Any], text_fields: list[str]) -> str:
    parts = []
    for f in text_fields:
        v = ex.get(f, "")
        if v is None:
            v = ""
        v = str(v).strip()
        if v:
            parts.append(v)
    return "\n".join(parts).strip()


def load_and_prepare(cfg: DataConfig) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    ds = load_dataset(cfg.dataset_name)

    if "train" not in ds:
        first = list(ds.keys())[0]
        ds = DatasetDict({"train": ds[first]})

    ds = ds.map(lambda ex: {"text": _build_text(ex, cfg.text_fields)}, desc="Build text")
    ds = ds.filter(lambda ex: ex.get("text", "").strip() != "" and ex.get(cfg.label_field) is not None)

    labels = sorted(set(ds["train"][cfg.label_field]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    ds = ds.map(lambda ex: {"label": label2id[ex[cfg.label_field]]}, desc="Encode labels")

    tmp = ds["train"].train_test_split(test_size=cfg.test_size, seed=cfg.seed)
    train_val = tmp["train"].train_test_split(
        test_size=cfg.val_size / (1.0 - cfg.test_size),
        seed=cfg.seed,
    )

    out = DatasetDict(train=train_val["train"], validation=train_val["test"], test=tmp["test"])
    return out, label2id, id2label