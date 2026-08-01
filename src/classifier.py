"""Document classifier: DeepEncoder (SAM→CLIP) CLS token + MLP head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from classes import NUM_CLASSES
from deepencoder import build_clip_l, build_sam_vit_b
from deepencoder_bundle import (
    DEFAULT_CHECKPOINT,
    DeepEncoder,
    load_deepseek_ocr_encoder_state_dict,
)


def build_mlp(
    in_dim: int,
    num_classes: int,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.1,
) -> nn.Sequential:
    hidden_dims = hidden_dims or [512]
    layers: list[nn.Module] = []
    dim = in_dim
    for h in hidden_dims:
        layers.extend(
            [
                nn.Linear(dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        dim = h
    layers.append(nn.Linear(dim, num_classes))
    return nn.Sequential(*layers)


class DocumentClassifier(nn.Module):
    """SAM → CLIP → CLS → MLP → class logits."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        mlp_hidden_dims: list[int] | None = None,
        mlp_dropout: float = 0.1,
        n_embed: int = 1280,
    ):
        super().__init__()
        self.encoder = DeepEncoder(n_embed=n_embed)
        self.cls_dim = 1024  # CLIP-L hidden size
        self.head = build_mlp(
            self.cls_dim,
            num_classes,
            hidden_dims=mlp_hidden_dims,
            dropout=mlp_dropout,
        )
        self.num_classes = num_classes

    def forward_cls(self, images: torch.Tensor) -> torch.Tensor:
        sam_feats = self.encoder.sam_model(images)
        clip_out = self.encoder.vision_model(images, sam_feats)
        return clip_out[:, 0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls = self.forward_cls(images)
        return self.head(cls)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def apply_train_from_zero_and_freeze(
    model: DocumentClassifier,
    *,
    train_from_zero_sam: bool,
    train_from_zero_clip: bool,
    freeze_sam: bool,
    freeze_clip: bool,
    checkpoint: str | Path | None = DEFAULT_CHECKPOINT,
) -> DocumentClassifier:
    """Load DeepSeek-OCR encoder weights selectively, then apply freeze flags.

    - ``train_from_zero.SAM/CLIP: true`` → keep random init for that tower (do not load).
    - When loading: only load the towers that are not from-zero.
    - ``freeze.*`` applies only when that tower was loaded (not from-zero).
      If from-zero, the tower is always trainable.
    - MLP head is always trainable. Projector stays unused for CLS classification.
    """
    load_sam = not train_from_zero_sam
    load_clip = not train_from_zero_clip

    if load_sam or load_clip:
        if checkpoint is None:
            raise ValueError("checkpoint path required when not training a tower from zero")
        state = load_deepseek_ocr_encoder_state_dict(checkpoint)
        filtered: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            if key.startswith("sam_model.") and load_sam:
                filtered[f"encoder.{key}"] = tensor
            elif key.startswith("vision_model.") and load_clip:
                filtered[f"encoder.{key}"] = tensor
            # projector intentionally skipped for CLS-head classifier
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        # Expected: projector + any tower we skipped + position_ids buffer
        _ = missing, unexpected

    if train_from_zero_sam:
        model.encoder.sam_model = build_sam_vit_b(checkpoint=None)
    if train_from_zero_clip:
        model.encoder.vision_model = build_clip_l()

    # Freeze / unfreeze towers
    sam_trainable = train_from_zero_sam or (not freeze_sam)
    clip_trainable = train_from_zero_clip or (not freeze_clip)

    for p in model.encoder.sam_model.parameters():
        p.requires_grad = sam_trainable
    for p in model.encoder.vision_model.parameters():
        p.requires_grad = clip_trainable
    for p in model.encoder.projector.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    return model


def build_document_classifier(model_cfg: dict[str, Any]) -> DocumentClassifier:
    """Build classifier from YAML ``model`` section."""
    tfz = model_cfg.get("train_from_zero") or {}
    freeze = model_cfg.get("freeze") or {}
    model = DocumentClassifier(
        num_classes=int(model_cfg.get("num_classes", NUM_CLASSES)),
        mlp_hidden_dims=list(model_cfg.get("mlp_hidden_dims") or [512]),
        mlp_dropout=float(model_cfg.get("mlp_dropout", 0.1)),
        n_embed=int(model_cfg.get("n_embed", 1280)),
    )
    ckpt = model_cfg.get("checkpoint")
    apply_train_from_zero_and_freeze(
        model,
        train_from_zero_sam=bool(tfz.get("SAM", False)),
        train_from_zero_clip=bool(tfz.get("CLIP", False)),
        freeze_sam=bool(freeze.get("SAM", True)),
        freeze_clip=bool(freeze.get("CLIP", False)),
        checkpoint=ckpt,
    )
    return model
