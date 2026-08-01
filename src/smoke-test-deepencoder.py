import sys
from pathlib import Path

import torch
from torchvision import transforms

ROOT = Path("/mnt/data/uncertainity_estimation")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docile"))  # if docile isn't installed

from deepencoder import build_sam_vit_b, build_clip_l

# --- 1) get a page image ---
from docile.dataset import CachingConfig, Dataset

DATASET_PATH = Path("/mnt/data/uncertainity_estimation/docile/data/docile")
dataset = Dataset("val", DATASET_PATH, cache_images=CachingConfig.DISK)
doc = dataset[1]

img = doc.page_image(page=0, image_size=(None, 1024))  # PIL

# --- 2) preprocess to SAM-style 1024 square ---
# Encoder assumes img_size=1024; no norm inside the module.
preprocess = transforms.Compose([
    transforms.Resize((640, 640)), # can change these to 512x512 or 1024x1024 as well to get 64 , 100 and 256 tokens at each resolution.
    transforms.ToTensor(),
])
x = preprocess(img.convert("RGB")).unsqueeze(0)  # (1, 3, 1024, 1024)

# --- 3) DeepEncoder: SAM (+16x compressor) -> CLIP-L (no patch embed) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# No checkpoint => random weights; fine to check plumbing/shapes
sam = build_sam_vit_b(checkpoint=None).to(device).eval()
clip = build_clip_l().to(device).eval()

with torch.inference_mode():
    x = x.to(device)
    sam_feats = sam(x)             # (1, 1024, 16, 16) — 256 vision tokens
    # pixel_values only used for batch size when patch_embeds is set
    clip_out = clip(x, sam_feats)  # (1, 257, 1024) — CLS + 256 tokens

print("sam:", type(sam_feats), sam_feats.shape, sam_feats.dtype)
print("clip:", type(clip_out), clip_out.shape, clip_out.dtype)
print(clip_out[0, 0, :8])  # peek at CLS features
