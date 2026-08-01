"""DocILE dataloaders for document classification experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from classes import ALLOWED_CLASSES, CLASS_TO_IDX, EXCLUDED_CLASSES, is_allowed_class

TrainMode = Literal["ceiling", "val", "val_plus_sampled"]


def load_json_ids(path: Path | str) -> list[str]:
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list of ids in {path}")
    return [str(x) for x in data]


def load_sampled_csv(path: Path | str) -> list[str]:
    """Load doc ids from CSV (column doc_id or first column) or one-id-per-line text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Empty sampled documents file: {path}")

    # Detect CSV with header
    lines = text.splitlines()
    first = lines[0].strip()
    ids: list[str] = []
    if "," in first or first.lower() in {"doc_id", "docid", "id"}:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV: {path}")
        header = [c.strip().lower() for c in rows[0]]
        start = 0
        col = 0
        if header[0] in {"doc_id", "docid", "id"} or "doc_id" in header:
            if "doc_id" in header:
                col = header.index("doc_id")
            elif "docid" in header:
                col = header.index("docid")
            elif "id" in header:
                col = header.index("id")
            start = 1
        for row in rows[start:]:
            if not row:
                continue
            ids.append(row[col].strip())
    else:
        ids = [
            ln.strip()
            for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        ]

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def read_document_type(ann_dir: Path, doc_id: str) -> str:
    path = ann_dir / f"{doc_id}.json"
    with path.open() as f:
        data = json.load(f)
    return str(data["metadata"]["document_type"])


def filter_allowed_ids(
    doc_ids: Sequence[str],
    ann_dir: Path,
    *,
    context: str,
) -> list[str]:
    kept: list[str] = []
    for doc_id in doc_ids:
        dtype = read_document_type(ann_dir, doc_id)
        if dtype in EXCLUDED_CLASSES or not is_allowed_class(dtype):
            raise ValueError(
                f"{context}: document {doc_id} has excluded/unknown class {dtype!r}"
            )
        kept.append(doc_id)
    return kept


def assert_no_holdout_overlap(
    train_ids: Sequence[str],
    holdout_ids: Sequence[str],
    *,
    context: str,
) -> None:
    holdout = set(holdout_ids)
    overlap = [i for i in train_ids if i in holdout]
    if overlap:
        raise ValueError(
            f"{context}: {len(overlap)} train ids overlap holdout "
            f"(e.g. {overlap[:5]})"
        )


def resolve_train_ids(data_cfg: dict[str, Any]) -> list[str]:
    """Resolve training document ids from ``data.train_mode``."""
    dataset_path = Path(data_cfg["dataset_path"])
    ann_dir = dataset_path / "annotations"
    holdout_ids = load_json_ids(data_cfg["holdout_path"])
    mode: TrainMode = data_cfg["train_mode"]

    if mode == "ceiling":
        pool = load_json_ids(data_cfg["train_pool_path"])
        val_ids = load_json_ids(dataset_path / "val.json")
        train_ids = list(dict.fromkeys([*pool, *val_ids]))
    elif mode == "val":
        train_ids = load_json_ids(dataset_path / "val.json")
    elif mode == "val_plus_sampled":
        csv_path = data_cfg.get("sampled_documents_csv")
        if not csv_path:
            raise ValueError(
                "data.sampled_documents_csv is required when train_mode=val_plus_sampled"
            )
        pool = set(load_json_ids(data_cfg["train_pool_path"]))
        sampled = load_sampled_csv(csv_path)
        if not sampled:
            raise ValueError(f"sampled_documents_csv is empty: {csv_path}")
        bad = [i for i in sampled if i not in pool]
        if bad:
            raise ValueError(
                f"sampled CSV contains {len(bad)} ids not in train_pool "
                f"(e.g. {bad[:5]})"
            )
        val_ids = load_json_ids(dataset_path / "val.json")
        train_ids = list(dict.fromkeys([*val_ids, *sampled]))
    else:
        raise ValueError(f"Unknown train_mode: {mode}")

    train_ids = filter_allowed_ids(train_ids, ann_dir, context=f"train({mode})")
    assert_no_holdout_overlap(train_ids, holdout_ids, context=f"train({mode})")
    return train_ids


def resolve_eval_ids(data_cfg: dict[str, Any]) -> list[str]:
    dataset_path = Path(data_cfg["dataset_path"])
    ann_dir = dataset_path / "annotations"
    holdout_ids = load_json_ids(data_cfg["holdout_path"])
    return filter_allowed_ids(holdout_ids, ann_dir, context="holdout")


class DocilePageDataset(Dataset):
    """Page-0 images + document_type labels for a fixed list of doc ids."""

    def __init__(
        self,
        dataset_path: Path | str,
        doc_ids: Sequence[str],
        *,
        image_size: int = 1024,
        page: int = 0,
        class_list: Sequence[str] | None = None,
    ):
        from docile.dataset import CachingConfig, Dataset as DocileDataset

        self.dataset_path = Path(dataset_path)
        self.doc_ids = list(doc_ids)
        self.image_size = int(image_size)
        self.page = int(page)
        self.class_list = list(class_list or ALLOWED_CLASSES)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}

        # Custom split name; pass explicit docids.
        self._docile = DocileDataset(
            split_name="custom",
            dataset_path=self.dataset_path,
            load_annotations=True,
            load_ocr=False,
            cache_images=CachingConfig.DISK,
            docids=self.doc_ids,
        )
        self._id_to_pos = {doc.docid: i for i, doc in enumerate(self._docile)}
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        doc_id = self.doc_ids[index]
        doc = self._docile[self._id_to_pos[doc_id]]
        dtype = doc.annotation.document_type
        if dtype not in self.class_to_idx:
            raise ValueError(f"Document {doc_id} has unexpected class {dtype!r}")
        # Render at ~200 DPI first, then resize to model input.
        img = doc.page_image(page=self.page, image_size=(None, None))
        x = self.transform(img.convert("RGB"))
        y = self.class_to_idx[dtype]
        return {
            "pixel_values": x,
            "labels": torch.tensor(y, dtype=torch.long),
            "doc_id": doc_id,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "doc_id": [b["doc_id"] for b in batch],
    }


def make_dataloader(
    dataset_path: Path | str,
    doc_ids: Sequence[str],
    *,
    image_size: int,
    page: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    class_list: Sequence[str] | None = None,
) -> DataLoader:
    ds = DocilePageDataset(
        dataset_path,
        doc_ids,
        image_size=image_size,
        page=page,
        class_list=class_list,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_train_eval_loaders(data_cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    train_ids = resolve_train_ids(data_cfg)
    eval_ids = resolve_eval_ids(data_cfg)
    common = dict(
        dataset_path=data_cfg["dataset_path"],
        image_size=int(data_cfg.get("image_size", 1024)),
        page=int(data_cfg.get("page", 0)),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        class_list=data_cfg.get("class_list") or ALLOWED_CLASSES,
    )
    train_loader = make_dataloader(
        doc_ids=train_ids,
        shuffle=True,
        **common,
    )
    eval_loader = make_dataloader(
        doc_ids=eval_ids,
        shuffle=False,
        **common,
    )
    print(
        f"train docs={len(train_ids)}  eval(holdout) docs={len(eval_ids)}  "
        f"mode={data_cfg['train_mode']}"
    )
    return train_loader, eval_loader
