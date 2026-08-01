#!/usr/bin/env python3
"""Summarize document_type class counts in DocILE annotated-trainval."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_doc_ids(split_path: Path) -> list[str]:
    with split_path.open() as f:
        return json.load(f)


def count_document_types(ann_dir: Path, doc_ids: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    missing = 0
    for doc_id in doc_ids:
        path = ann_dir / f"{doc_id}.json"
        if not path.exists():
            missing += 1
            continue
        with path.open() as f:
            data = json.load(f)
        doc_type = data["metadata"]["document_type"]
        counts[doc_type] += 1
    if missing:
        print(f"  warning: {missing} annotation files missing")
    return counts


def print_counts(name: str, counts: Counter[str], total: int) -> None:
    print(f"\n{name} ({total} documents)")
    print("-" * 40)
    print(f"{'class':<20} {'count':>8} {'pct':>8}")
    for cls, n in counts.most_common():
        pct = 100.0 * n / total if total else 0.0
        print(f"{cls:<20} {n:>8} {pct:>7.1f}%")
    print(f"{'TOTAL':<20} {sum(counts.values()):>8}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("docile/data/docile"),
        help="Path to unzipped annotated-trainval root (has train.json, val.json, annotations/)",
    )
    args = parser.parse_args()

    root = args.dataset_path
    ann_dir = root / "annotations"

    train_ids = load_doc_ids(root / "train.json")
    val_ids = load_doc_ids(root / "val.json")

    train_counts = count_document_types(ann_dir, train_ids)
    val_counts = count_document_types(ann_dir, val_ids)
    all_counts = train_counts + val_counts

    print_counts("train", train_counts, len(train_ids))
    print_counts("val", val_counts, len(val_ids))
    print_counts("train+val", all_counts, len(train_ids) + len(val_ids))

    # Reminder of assignment split rename
    print("\nAssignment remap reminder:")
    print("  val.json   -> labeled train  (new_train.json)")
    print("  train.json -> unlabeled pool (unlabeled.json)")
    print_counts("assignment labeled train (val)", val_counts, len(val_ids))
    print_counts("assignment unlabeled pool (train)", train_counts, len(train_ids))


if __name__ == "__main__":
    main()

# train (5180 documents)
# ----------------------------------------
# class                   count      pct
# tax_invoice              3512    67.8%
# order                    1333    25.7%
# purchase_order            117     2.3%
# receipt                    95     1.8%
# sales_order                54     1.0%
# proforma                   28     0.5%
# credit_note                23     0.4%
# utility_bill               12     0.2%
# debit_note                  6     0.1%
# TOTAL                    5180

# val (500 documents)
# ----------------------------------------
# class                   count      pct
# tax_invoice               338    67.6%
# order                     107    21.4%
# sales_order                21     4.2%
# receipt                    21     4.2%
# purchase_order             11     2.2%
# proforma                    1     0.2%
# credit_note                 1     0.2%
# TOTAL                     500

# train+val (5680 documents)
# ----------------------------------------
# class                   count      pct
# tax_invoice              3850    67.8%
# order                    1440    25.4%
# purchase_order            128     2.3%
# receipt                   116     2.0%
# sales_order                75     1.3%
# proforma                   29     0.5%
# credit_note                24     0.4%
# utility_bill               12     0.2%
# debit_note                  6     0.1%
# TOTAL                    5680

# Assignment remap reminder:
#   val.json   -> labeled train  (new_train.json)
#   train.json -> unlabeled pool (unlabeled.json)

# assignment labeled train (val) (500 documents)
# ----------------------------------------
# class                   count      pct
# tax_invoice               338    67.6%
# order                     107    21.4%
# sales_order                21     4.2%
# receipt                    21     4.2%
# purchase_order             11     2.2%
# proforma                    1     0.2%
# credit_note                 1     0.2%
# TOTAL                     500

# assignment unlabeled pool (train) (5180 documents)
# ----------------------------------------
# class                   count      pct
# tax_invoice              3512    67.8%
# order                    1333    25.7%
# purchase_order            117     2.3%
# receipt                    95     1.8%
# sales_order                54     1.0%
# proforma                   28     0.5%
# credit_note                23     0.4%
# utility_bill               12     0.2%
# debit_note                  6     0.1%
# TOTAL                    5180