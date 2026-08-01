"""Pure-PyTorch DeepEncoder wrapper + DeepSeek-OCR weight loader.

Assembles SAM ViT-B + CLIP-L + linear projector and loads only the matching
tensors from a DeepSeek-OCR safetensors checkpoint (no Transformers required).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from easydict import EasyDict as Dict
from safetensors import safe_open

from deepencoder import MlpProjector, build_clip_l, build_sam_vit_b

DEFAULT_CHECKPOINT = Path("/mnt/data/DeepSeek-OCR/model-00001-of-000001.safetensors")

# Buffers regenerated at init; not stored in the DeepSeek-OCR shard.
_EXPECTED_MISSING = frozenset({"vision_model.embeddings.position_ids"})

_ENCODER_PREFIXES = (
    "model.sam_model.",
    "model.vision_model.",
    "model.projector.",
)


class DeepEncoder(nn.Module):
    """SAM → CLIP → concat(clip[1:], sam) → linear projector (2048 → n_embed)."""

    def __init__(self, n_embed: int = 1280):
        super().__init__()
        self.sam_model = build_sam_vit_b(checkpoint=None)
        self.vision_model = build_clip_l()
        self.projector = MlpProjector(
            Dict(projector_type="linear", input_dim=2048, n_embed=n_embed)
        )
        self.n_embed = n_embed

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: `(B, 3, H, W)` float tensor in roughly `[0, 1]` (no mean/std
                inside the module; match DeepSeek-OCR preprocessing if needed).

        Returns:
            Vision tokens `(B, N, n_embed)` where `N = (H/64) * (W/64)`.
        """
        sam_feats = self.sam_model(images)
        clip_out = self.vision_model(images, sam_feats)
        fused = torch.cat(
            (clip_out[:, 1:], sam_feats.flatten(2).permute(0, 2, 1)),
            dim=-1,
        )
        return self.projector(fused)

    def freeze(self) -> DeepEncoder:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self

    def unfreeze(self) -> DeepEncoder:
        for p in self.parameters():
            p.requires_grad = True
        self.train()
        return self


def _iter_encoder_tensors(
    checkpoint: str | Path,
    device: str = "cpu",
) -> Iterable[tuple[str, torch.Tensor]]:
    path = str(checkpoint)
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():
            if not key.startswith(_ENCODER_PREFIXES):
                continue
            # Strip leading "model." so keys match DeepEncoder.state_dict().
            yield key[len("model.") :], f.get_tensor(key)


def load_deepseek_ocr_encoder(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    n_embed: int = 1280,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> DeepEncoder:
    """Build a DeepEncoder and load SAM / CLIP / projector weights from a shard.

    Only encoder-related tensors are read from disk; LLM weights are skipped.
    """
    model = DeepEncoder(n_embed=n_embed)
    state = dict(_iter_encoder_tensors(checkpoint, device="cpu"))
    if not state:
        raise FileNotFoundError(
            f"No sam_model/vision_model/projector tensors found in {checkpoint}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k not in _EXPECTED_MISSING]
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys: {unexpected[:20]}")
    if missing and strict:
        raise RuntimeError(f"Missing encoder keys: {missing[:20]}")

    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device)


def load_deepseek_ocr_encoder_state_dict(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
) -> dict[str, torch.Tensor]:
    """Return a state_dict suitable for `DeepEncoder.load_state_dict`."""
    return dict(_iter_encoder_tensors(checkpoint, device="cpu"))
