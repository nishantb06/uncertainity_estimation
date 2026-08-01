"""Canonical DocILE document_type labels for this project (val-aligned, 7-way)."""

from __future__ import annotations

# Val classes only — used for training, holdout, and evaluation.
ALLOWED_CLASSES: list[str] = [
    "credit_note",
    "order",
    "proforma",
    "purchase_order",
    "receipt",
    "sales_order",
    "tax_invoice",
]

# Present in train.json but excluded from all experiments.
EXCLUDED_CLASSES: list[str] = [
    "debit_note",
    "utility_bill",
]

CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(ALLOWED_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES: int = len(ALLOWED_CLASSES)


def is_allowed_class(document_type: str) -> bool:
    return document_type in CLASS_TO_IDX
