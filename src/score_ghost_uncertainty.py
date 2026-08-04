#!/usr/bin/env python3
"""Score train_pool docs with multi-resolution ghost-gradient alignment.

Loads a Lightning DocumentClassifier checkpoint, runs encoder under no_grad and
head with grad at 1024 / 640 / 512, captures ghost (a, δ) factors on all head
Linear layers, and writes pairwise cosine / uncertainty CSVs + histograms.
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
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCILE = ROOT / "docile"
for path in (SRC, DOCILE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from classifier import DocumentClassifier  # noqa: E402
from dataloaders import load_json_ids  # noqa: E402
from ghost_gradients import (  # noqa: E402
    GhostGradientCapture,
    load_head_preconditioners,
    multi_layer_cosine,
)
from score_uncertainty import (  # noqa: E402
    RawPageDataset,
    collate_raw,
    images_to_batch,
    load_classifier,
    load_config_near_checkpoint,
)

RESOLUTIONS = (1024, 640, 512)
PAIR_COLUMNS = [
    ("ghost_cos_1024_640", 1024, 640),
    ("ghost_cos_1024_512", 1024, 512),
    ("ghost_cos_640_512", 640, 512),
]
METRIC_COLUMNS = [
    "ghost_cos_1024_640",
    "ghost_cos_1024_512",
    "ghost_cos_640_512",
    "ghost_avg_cos",
    "ghost_uncertainty",
]


def capture_resolution_factors(
    model: DocumentClassifier,
    images: list[Any],
    res: int,
    device: torch.device,
    capture: GhostGradientCapture,
    preconditioners: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """One forward+backward at resolution ``res``; return ghost factors."""
    capture.clear()
    model.zero_grad(set_to_none=True)

    x = images_to_batch(images, res, device)
    with torch.no_grad():
        cls = model.forward_cls(x)
    cls = cls.detach().requires_grad_(True)

    # Dropout off for stable scores; hooks still fire in eval mode.
    model.head.eval()
    logits = model.head(cls)
    pred = logits.argmax(dim=-1)
    loss = F.cross_entropy(logits, pred)
    loss.backward()

    return capture.factors(preconditioners)


def score_batch_ghost(
    model: DocumentClassifier,
    images: list[Any],
    device: torch.device,
    capture: GhostGradientCapture,
    preconditioners: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]],
) -> dict[str, torch.Tensor]:
    factors_by_res: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for res in RESOLUTIONS:
        factors_by_res[res] = capture_resolution_factors(
            model, images, res, device, capture, preconditioners
        )

    out: dict[str, torch.Tensor] = {}
    pair_vals: list[torch.Tensor] = []
    for col, r1, r2 in PAIR_COLUMNS:
        cos = multi_layer_cosine(factors_by_res[r1], factors_by_res[r2])
        out[col] = cos
        pair_vals.append(cos)

    avg = torch.stack(pair_vals, dim=0).mean(dim=0)
    out["ghost_avg_cos"] = avg
    out["ghost_uncertainty"] = 1.0 - avg
    return out


def save_histograms(df: pd.DataFrame, out_dir: Path, stem: str) -> list[str]:
    paths: list[str] = []
    for col in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[col].to_numpy(), bins=50, color="#6b4c9a", edgecolor="white")
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
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--doc-ids-json",
        type=Path,
        default=ROOT / "splits" / "train_pool.json",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--top-k", type=int, default=1500)
    p.add_argument(
        "--rank-by",
        type=str,
        default="ghost_uncertainty",
        choices=METRIC_COLUMNS,
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "uncertainty_scores",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--no-precond",
        action="store_true",
        help="Ignore AdamW exp_avg_sq in the checkpoint (identity preconditioner)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    cfg = load_config_near_checkpoint(checkpoint, args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    dataset_path = Path(data_cfg["dataset_path"])
    page = int(data_cfg.get("page", 0))

    doc_ids = load_json_ids(args.doc_ids_json)
    if not doc_ids:
        raise ValueError(f"No document ids in {args.doc_ids_json}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"checkpoint: {checkpoint}")
    print(f"docs: {len(doc_ids)} from {args.doc_ids_json}")
    print(f"device: {device}")

    # Load weights
    model = load_classifier(checkpoint, model_cfg, device)
    model.eval()
    model.head.eval()

    # Optional AdamW preconditioners from optimizer state in the same ckpt
    preconditioners: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    if not args.no_precond:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        opt_states = blob.get("optimizer_states")
        preconditioners = load_head_preconditioners(opt_states, model, module_prefix="head")
        if preconditioners:
            print(f"loaded AdamW preconditioners for {len(preconditioners)} head Linears")
        else:
            print("no usable AdamW state — using identity preconditioner")

    capture = GhostGradientCapture(model, module_prefix="head")
    if not capture.hooks:
        capture.remove()
        raise RuntimeError("No Linear layers found under model.head")

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
    try:
        for batch in tqdm(loader, desc="ghost-scoring"):
            metrics = score_batch_ghost(
                model, batch["images"], device, capture, preconditioners
            )
            bsz = len(batch["doc_id"])
            for i in range(bsz):
                row = {"doc_id": batch["doc_id"][i]}
                for col in METRIC_COLUMNS:
                    row[col] = float(metrics[col][i].item())
                rows.append(row)
    finally:
        capture.remove()

    df = pd.DataFrame(rows)
    assert len(df) == len(doc_ids)

    run_name = checkpoint.parent.name
    stem = re.sub(r"[^\w.\-]+", "_", checkpoint.stem)
    out_dir = args.out_dir.resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = out_dir / f"{stem}_ghost_information.csv"
    df.to_csv(full_csv, index=False)

    ranked = df.sort_values(args.rank_by, ascending=False, kind="mergesort")
    top = ranked.head(args.top_k)[["doc_id"]]
    top_csv = out_dir / f"{stem}_top{args.top_k}_by_{args.rank_by}.csv"
    top.to_csv(top_csv, index=False)

    hist_paths = save_histograms(df, out_dir, stem)

    summary = {
        "checkpoint": str(checkpoint),
        "doc_ids_json": str(args.doc_ids_json.resolve()),
        "n_docs": len(df),
        "resolutions": list(RESOLUTIONS),
        "rank_by": args.rank_by,
        "top_k": args.top_k,
        "used_preconditioners": bool(preconditioners),
        "outputs": {
            "ghost_information_csv": str(full_csv),
            "top_k_csv": str(top_csv),
            "histograms": hist_paths,
        },
        "metric_mean": {c: float(df[c].mean()) for c in METRIC_COLUMNS},
        "metric_std": {c: float(df[c].std(ddof=0)) for c in METRIC_COLUMNS},
    }
    summary_path = out_dir / f"{stem}_ghost_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote {full_csv}")
    print(f"wrote {top_csv}")
    print(f"wrote {summary_path}")
    for hp in hist_paths:
        print(f"wrote {hp}")


if __name__ == "__main__":
    main()
