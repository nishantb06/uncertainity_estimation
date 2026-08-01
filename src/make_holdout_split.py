#!/usr/bin/env python3
"""Create stratified 180-doc holdout from DocILE train (val classes only)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from classes import ALLOWED_CLASSES, EXCLUDED_CLASSES


def load_ids(path: Path) -> list[str]:
    with path.open() as f:
        return [str(x) for x in json.load(f)]


def document_type(ann_dir: Path, doc_id: str) -> str:
    with (ann_dir / f"{doc_id}.json").open() as f:
        return str(json.load(f)["metadata"]["document_type"])


def stratified_sample(
    by_class: dict[str, list[str]],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Sample ``n`` ids with approx class proportions and ≥1 per class."""
    classes = [c for c in ALLOWED_CLASSES if by_class.get(c)]
    missing = [c for c in ALLOWED_CLASSES if not by_class.get(c)]
    if missing:
        raise RuntimeError(f"No eligible train docs for classes: {missing}")

    total = sum(len(by_class[c]) for c in classes)
    if n > total:
        raise ValueError(f"Cannot sample n={n} from only {total} eligible docs")
    if n < len(classes):
        raise ValueError(f"n={n} too small to include all {len(classes)} classes")

    # Initial allocation: max(1, round(prop * n))
    alloc: dict[str, int] = {}
    for c in classes:
        prop = len(by_class[c]) / total
        alloc[c] = max(1, int(round(prop * n)))

    # Cap by availability
    for c in classes:
        alloc[c] = min(alloc[c], len(by_class[c]))

    # Adjust to exactly n
    def current_sum() -> int:
        return sum(alloc.values())

    # Shrink if over
    while current_sum() > n:
        # Prefer reducing classes with largest surplus over proportional target
        candidates = [
            c
            for c in classes
            if alloc[c] > 1
        ]
        if not candidates:
            break
        c = max(
            candidates,
            key=lambda x: alloc[x] - (len(by_class[x]) / total) * n,
        )
        alloc[c] -= 1

    # Grow if under
    while current_sum() < n:
        candidates = [
            c for c in classes if alloc[c] < len(by_class[c])
        ]
        if not candidates:
            break
        c = min(
            candidates,
            key=lambda x: alloc[x] - (len(by_class[x]) / total) * n,
        )
        alloc[c] += 1

    if current_sum() != n:
        raise RuntimeError(
            f"Failed to allocate exactly n={n} (got {current_sum()}): {alloc}"
        )

    sampled: list[str] = []
    for c in classes:
        pool = list(by_class[c])
        rng.shuffle(pool)
        sampled.extend(pool[: alloc[c]])
    rng.shuffle(sampled)
    return sampled


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=root / "docile" / "data" / "docile",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "splits",
    )
    parser.add_argument("--n", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ann_dir = args.dataset_path / "annotations"
    train_ids = load_ids(args.dataset_path / "train.json")

    by_class: dict[str, list[str]] = defaultdict(list)
    dropped: list[dict[str, str]] = []
    for doc_id in train_ids:
        dtype = document_type(ann_dir, doc_id)
        if dtype in EXCLUDED_CLASSES:
            dropped.append({"doc_id": doc_id, "document_type": dtype})
            continue
        if dtype not in ALLOWED_CLASSES:
            dropped.append({"doc_id": doc_id, "document_type": dtype})
            continue
        by_class[dtype].append(doc_id)

    eligible = sum(len(v) for v in by_class.values())
    print(f"train={len(train_ids)}  eligible={eligible}  dropped={len(dropped)}")
    for c in ALLOWED_CLASSES:
        print(f"  {c}: {len(by_class[c])}")

    rng = random.Random(args.seed)
    holdout = stratified_sample(by_class, args.n, rng)
    holdout_set = set(holdout)

    train_pool: list[str] = []
    for c in ALLOWED_CLASSES:
        for doc_id in by_class[c]:
            if doc_id not in holdout_set:
                train_pool.append(doc_id)

    holdout_counts = Counter(
        document_type(ann_dir, i) for i in holdout
    )
    pool_counts = Counter(document_type(ann_dir, i) for i in train_pool)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = args.out_dir / "holdout_180.json"
    pool_path = args.out_dir / "train_pool.json"
    dropped_path = args.out_dir / "dropped_excluded_classes.json"
    meta_path = args.out_dir / "holdout_180_meta.json"

    holdout_path.write_text(json.dumps(holdout, indent=2) + "\n")
    pool_path.write_text(json.dumps(train_pool, indent=2) + "\n")
    dropped_path.write_text(json.dumps(dropped, indent=2) + "\n")
    meta = {
        "seed": args.seed,
        "n_holdout": len(holdout),
        "n_train_pool": len(train_pool),
        "n_dropped": len(dropped),
        "allowed_classes": ALLOWED_CLASSES,
        "excluded_classes": EXCLUDED_CLASSES,
        "holdout_counts": dict(holdout_counts),
        "train_pool_counts": dict(pool_counts),
        "eligible_train_counts": {c: len(by_class[c]) for c in ALLOWED_CLASSES},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {holdout_path} ({len(holdout)})")
    print(f"wrote {pool_path} ({len(train_pool)})")
    print(f"wrote {dropped_path} ({len(dropped)})")
    print(f"wrote {meta_path}")
    print("holdout_counts:", dict(holdout_counts))


if __name__ == "__main__":
    main()
