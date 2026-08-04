#!/usr/bin/env python3
"""Score train_pool documents with multi-resolution entropy and JSD.

Loads a Lightning DocumentClassifier checkpoint, runs batched inference at
512 / 640 / 1024, and writes:
  - {stem}_entropy_information.csv
  - {stem}_top{K}_by_{rank_by}.csv
  - histogram PNGs
  - {stem}_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCILE = ROOT / "docile"
for path in (SRC, DOCILE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from classifier import DocumentClassifier  # noqa: E402
from classes import ALLOWED_CLASSES  # noqa: E402
from dataloaders import load_json_ids  # noqa: E402

RESOLUTIONS = (512, 640, 1024)
METRIC_COLUMNS = [
    "entropy_1024",
    "entropy_640",
    "entropy_512",
    "entropy_mean_logits",
    "jsd_1024_512",
    "jsd_1024_640",
    "jsd_512_640",
]


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats from logits. Shape (B, C) -> (B,)."""
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen–Shannon divergence with natural log. Shape (B, C) -> (B,)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (
        q * (q.log() - m.log())
    ).sum(dim=-1)


def remap_lightning_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip Lightning / torch.compile prefixes to match DocumentClassifier."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        k = key
        for prefix in (
            "_compiled_model._orig_mod.",
            "model._orig_mod.",
            "_compiled_model.",
            "model.",
        ):
            if k.startswith(prefix):
                k = k[len(prefix) :]
                break
        # Compiled modules sometimes nest _orig_mod again
        k = k.replace("._orig_mod.", ".")
        out[k] = value
    return out


def load_config_near_checkpoint(
    checkpoint: Path,
    config_override: Path | None,
) -> dict[str, Any]:
    if config_override is not None:
        with config_override.open() as f:
            return yaml.safe_load(f)
    sibling = checkpoint.parent / "config.yaml"
    if sibling.exists():
        with sibling.open() as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(
        f"No config.yaml next to {checkpoint}; pass --config explicitly"
    )


def build_model_from_config(model_cfg: dict[str, Any]) -> DocumentClassifier:
    """Build empty classifier (weights come from the Lightning ckpt)."""
    return DocumentClassifier(
        num_classes=int(model_cfg.get("num_classes", len(ALLOWED_CLASSES))),
        mlp_hidden_dims=list(model_cfg.get("mlp_hidden_dims") or [512]),
        mlp_dropout=float(model_cfg.get("mlp_dropout", 0.1)),
        n_embed=int(model_cfg.get("n_embed", 1280)),
    )


def load_classifier(
    checkpoint: Path,
    model_cfg: dict[str, Any],
    device: torch.device,
) -> DocumentClassifier:
    model = build_model_from_config(model_cfg)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in blob:
        raise KeyError(f"Checkpoint missing state_dict: {checkpoint}")
    state = remap_lightning_state_dict(blob["state_dict"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Projector may be unused / random; head + towers should load.
    bad_missing = [
        m
        for m in missing
        if not m.startswith("encoder.projector.")
        and "position_ids" not in m
    ]
    if bad_missing:
        raise RuntimeError(
            f"Missing unexpected keys when loading {checkpoint}: {bad_missing[:20]}"
        )
    if unexpected:
        print(f"warning: unexpected keys ignored ({len(unexpected)}), e.g. {unexpected[:5]}")
    return model.to(device).eval()


class RawPageDataset(Dataset):
    """Render each page once as a PIL RGB image (no fixed resize)."""

    def __init__(
        self,
        dataset_path: Path,
        doc_ids: list[str],
        *,
        page: int = 0,
    ):
        from docile.dataset import CachingConfig, Dataset as DocileDataset

        self.doc_ids = list(doc_ids)
        self.page = int(page)
        self._docile = DocileDataset(
            split_name="custom",
            dataset_path=dataset_path,
            load_annotations=True,
            load_ocr=False,
            cache_images=CachingConfig.DISK,
            docids=self.doc_ids,
        )
        self._id_to_pos = {doc.docid: i for i, doc in enumerate(self._docile)}

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        doc_id = self.doc_ids[index]
        doc = self._docile[self._id_to_pos[doc_id]]
        img = doc.page_image(page=self.page, image_size=(None, None)).convert("RGB")
        return {"image": img, "doc_id": doc_id}


def collate_raw(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [b["image"] for b in batch],
        "doc_id": [b["doc_id"] for b in batch],
    }


def images_to_batch(
    images: list[Any],
    size: int,
    device: torch.device,
) -> torch.Tensor:
    tensors = [TF.to_tensor(TF.resize(img, [size, size])) for img in images]
    return torch.stack(tensors, dim=0).to(device)


@torch.inference_mode()
def score_batch(
    model: DocumentClassifier,
    images: list[Any],
    device: torch.device,
    use_amp: bool,
) -> dict[str, torch.Tensor]:
    logits_by_res: dict[int, torch.Tensor] = {}
    for res in RESOLUTIONS:
        x = images_to_batch(images, res, device)
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
        else:
            logits = model(x)
        logits_by_res[res] = logits.float()

    z1024 = logits_by_res[1024]
    z640 = logits_by_res[640]
    z512 = logits_by_res[512]
    z_mean = (z1024 + z640 + z512) / 3.0

    p1024 = F.softmax(z1024, dim=-1)
    p640 = F.softmax(z640, dim=-1)
    p512 = F.softmax(z512, dim=-1)

    return {
        "entropy_1024": entropy_from_logits(z1024),
        "entropy_640": entropy_from_logits(z640),
        "entropy_512": entropy_from_logits(z512),
        "entropy_mean_logits": entropy_from_logits(z_mean),
        "jsd_1024_512": jsd(p1024, p512),
        "jsd_1024_640": jsd(p1024, p640),
        "jsd_512_640": jsd(p512, p640),
        "pred_1024": z1024.argmax(dim=-1),
        "pred_640": z640.argmax(dim=-1),
        "pred_512": z512.argmax(dim=-1),
    }


def save_histograms(df: pd.DataFrame, out_dir: Path, stem: str) -> list[str]:
    paths: list[str] = []
    for col in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[col].to_numpy(), bins=50, color="#3b6ea5", edgecolor="white")
        ax.set_title(f"{col} (n={len(df)})")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        fig.tight_layout()
        path = out_dir / f"{stem}_hist_{col}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Lightning .ckpt path (e.g. .../last.ckpt)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML; defaults to config.yaml next to the checkpoint",
    )
    p.add_argument(
        "--doc-ids-json",
        type=Path,
        default=ROOT / "splits" / "train_pool.json",
        help="JSON list of document ids to score (default: train_pool)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--top-k", type=int, default=1500)
    p.add_argument(
        "--rank-by",
        type=str,
        default="entropy_mean_logits",
        choices=METRIC_COLUMNS,
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "uncertainty_scores",
        help="Base output directory; a run subfolder is created from the ckpt parent name",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    cfg = load_config_near_checkpoint(checkpoint, args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    class_list = list(data_cfg.get("class_list") or ALLOWED_CLASSES)
    dataset_path = Path(data_cfg["dataset_path"])
    page = int(data_cfg.get("page", 0))

    doc_ids = load_json_ids(args.doc_ids_json)
    if not doc_ids:
        raise ValueError(f"No document ids in {args.doc_ids_json}")

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = (not args.no_amp) and device.type == "cuda"

    print(f"checkpoint: {checkpoint}")
    print(f"docs: {len(doc_ids)} from {args.doc_ids_json}")
    print(f"device: {device}  amp={use_amp}")

    model = load_classifier(checkpoint, model_cfg, device)

    ds = RawPageDataset(dataset_path, doc_ids, page=page)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_raw,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="scoring"):
        metrics = score_batch(model, batch["images"], device, use_amp)
        bsz = len(batch["doc_id"])
        for i in range(bsz):
            row = {"doc_id": batch["doc_id"][i]}
            for col in METRIC_COLUMNS:
                row[col] = float(metrics[col][i].item())
            for res, key in (
                (1024, "pred_1024"),
                (640, "pred_640"),
                (512, "pred_512"),
            ):
                idx = int(metrics[key][i].item())
                row[f"pred_{res}"] = class_list[idx] if idx < len(class_list) else str(idx)
            rows.append(row)

    df = pd.DataFrame(rows)
    # Preserve input order then stable sort for top-k
    assert len(df) == len(doc_ids)

    run_name = checkpoint.parent.name
    stem = checkpoint.stem
    # Sanitize stem for filesystem
    stem = re.sub(r"[^\w.\-]+", "_", stem)
    out_dir = args.out_dir.resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = out_dir / f"{stem}_entropy_information.csv"
    df.to_csv(full_csv, index=False)

    ranked = df.sort_values(args.rank_by, ascending=False, kind="mergesort")
    top = ranked.head(args.top_k)[["doc_id"]]
    top_csv = out_dir / f"{stem}_top{args.top_k}_by_{args.rank_by}.csv"
    top.to_csv(top_csv, index=False)

    hist_paths = save_histograms(df, out_dir, stem)

    summary = {
        "checkpoint": str(checkpoint),
        "config_source": str(
            args.config.resolve()
            if args.config is not None
            else checkpoint.parent / "config.yaml"
        ),
        "doc_ids_json": str(args.doc_ids_json.resolve()),
        "n_docs": len(df),
        "resolutions": list(RESOLUTIONS),
        "rank_by": args.rank_by,
        "top_k": args.top_k,
        "outputs": {
            "entropy_information_csv": str(full_csv),
            "top_k_csv": str(top_csv),
            "histograms": hist_paths,
        },
        "metric_mean": {c: float(df[c].mean()) for c in METRIC_COLUMNS},
        "metric_std": {c: float(df[c].std(ddof=0)) for c in METRIC_COLUMNS},
    }
    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote {full_csv}")
    print(f"wrote {top_csv}")
    print(f"wrote {summary_path}")
    for hp in hist_paths:
        print(f"wrote {hp}")


if __name__ == "__main__":
    main()
