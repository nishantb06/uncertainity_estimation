"""Smoke-test DeepEncoder with optional DeepSeek-OCR pretrained weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision import transforms

ROOT = Path("/mnt/data/uncertainity_estimation")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docile"))

from deepencoder_bundle import DEFAULT_CHECKPOINT, DeepEncoder, load_deepseek_ocr_encoder

# Resolution → expected token count (H/64 * W/64)
RESOLUTION_TOKENS = {
    512: 64,
    640: 100,
    1024: 256,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="DeepSeek-OCR safetensors shard (encoder tensors only are loaded)",
    )
    p.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip checkpoint load (random init; shape check only)",
    )
    p.add_argument("--size", type=int, default=1024, choices=sorted(RESOLUTION_TOKENS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    expected_tokens = RESOLUTION_TOKENS[args.size]

    from docile.dataset import CachingConfig, Dataset

    dataset = Dataset(
        "val",
        ROOT / "docile" / "data" / "docile",
        cache_images=CachingConfig.DISK,
    )
    img = dataset[1].page_image(page=0, image_size=(None, 1024))

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
        ]
    )
    x = preprocess(img.convert("RGB")).unsqueeze(0)

    if args.no_weights:
        model = DeepEncoder().to(args.device).eval()
        print("weights: random ( --no-weights )")
    else:
        dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        model = load_deepseek_ocr_encoder(
            args.checkpoint,
            device=args.device,
            dtype=dtype,
        ).eval()
        print(f"weights: {args.checkpoint}")

    with torch.inference_mode():
        x = x.to(device=args.device, dtype=next(model.parameters()).dtype)
        tokens = model(x)

    print("tokens:", tuple(tokens.shape), tokens.dtype)
    assert tokens.shape == (1, expected_tokens, model.n_embed), (
        f"expected (1, {expected_tokens}, {model.n_embed}), got {tuple(tokens.shape)}"
    )
    print("ok: projector output shape matches", args.size, "→", expected_tokens, "tokens")
    print("peek:", tokens[0, 0, :8].float().cpu())


if __name__ == "__main__":
    main()
