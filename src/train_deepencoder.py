"""Minimal pure-PyTorch training loop using pretrained DeepEncoder features.

By default the encoder is frozen and a small linear head is trained on a dummy
regression target (mean of vision tokens). Swap the head / loss for a real task.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path("/mnt/data/uncertainity_estimation")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docile"))

from deepencoder_bundle import DEFAULT_CHECKPOINT, load_deepseek_ocr_encoder


class DocilePageDataset(Dataset):
    """Tiny DocILE page loader: returns a square RGB tensor in [0, 1]."""

    def __init__(self, split: str, size: int, max_docs: int | None = 8):
        from docile.dataset import CachingConfig, Dataset as DocileDataset

        self.size = size
        self.ds = DocileDataset(
            split,
            ROOT / "docile" / "data" / "docile",
            cache_images=CachingConfig.DISK,
        )
        n = len(self.ds) if max_docs is None else min(max_docs, len(self.ds))
        self.ids = list(range(n))
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        doc = self.ds[self.ids[idx]]
        img = doc.page_image(page=0, image_size=(None, 1024)).convert("RGB")
        image = self.preprocess(img)
        # Dummy target: scalar that the head should regress from pooled tokens.
        target = image.mean().view(1)
        return {"image": image, "target": target}


class EncoderWithHead(nn.Module):
    def __init__(self, encoder: nn.Module, n_embed: int, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            self.encoder.freeze()
        self.head = nn.Linear(n_embed, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if next(self.encoder.parameters()).requires_grad:
            tokens = self.encoder(images)
        else:
            with torch.no_grad():
                tokens = self.encoder(images)
            tokens = tokens.detach()
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--size", type=int, default=512, help="Square input size (512/640/1024)")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-docs", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    encoder = load_deepseek_ocr_encoder(
        args.checkpoint,
        device=device,
        dtype=dtype,
    )
    model = EncoderWithHead(
        encoder,
        n_embed=encoder.n_embed,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    # Keep head in fp32 for stable Adam updates even when encoder is bf16.
    model.head = model.head.to(dtype=torch.float32)

    ds = DocilePageDataset("val", size=args.size, max_docs=args.max_docs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters; pass --no-freeze-encoder?")
    opt = torch.optim.AdamW(params, lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    # Encoder stays eval when frozen (BatchNorm-free, but keeps dropout off if any).
    if args.freeze_encoder:
        model.encoder.eval()

    step = 0
    while step < args.steps:
        for batch in loader:
            images = batch["image"].to(device=device, dtype=dtype)
            target = batch["target"].to(device=device, dtype=torch.float32)

            pred = model(images).float()
            loss = loss_fn(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            n_train = sum(p.numel() for p in params)
            print(
                f"step {step}/{args.steps}  loss={loss.item():.6f}  "
                f"trainable_params={n_train}  freeze_encoder={args.freeze_encoder}"
            )
            if step >= args.steps:
                break

    print("done")


if __name__ == "__main__":
    main()
