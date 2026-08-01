Good catch — but this isn't a bug, it's expected: **256 is the correct token count for the resolution you're testing at.**

## The 64/100 numbers are for different (smaller) resolution modes

Recall Table 1 from the paper:

| Mode | Resolution | Tokens |
|---|---|---|
| Tiny | 512×512 | 64 |
| Small | 640×640 | 100 |
| **Base** | **1024×1024** | **256** |
| Large | 1280×1280 | 400 |

Your script resizes the input to **1024×1024**:
```python
preprocess = transforms.Compose([
    transforms.Resize((1024, 1024)),
    ...
])
```
That's exactly **Base mode**, and Base mode is defined as **256 tokens**. So `clip_out.shape == (1, 257, 1024)` (256 patch tokens + 1 CLS) is correct — you're not seeing a discrepancy, you're just running the 256-token configuration, not the 64 or 100-token ones.

## Why resolution controls token count here

Both SAM and CLIP's grid sizes are direct functions of input resolution ÷ patch size:
- SAM: `img_size / patch_size` → with the two stride-2 conv downsamples in `net_2`/`net_3`, a 1024×1024 input → 64×64 → 32×32 → **16×16 = 256 spatial positions**
- CLIP: config is `image_size=224, patch_size=14` → natively 16×16 = 256 patches, which is why it lines up with SAM's output size with no interpolation needed at this specific resolution

If you want to reproduce the **64-token (Tiny)** or **100-token (Small)** configurations from the paper, you'd resize the input differently:

- **Tiny (64 tokens)**: resize input to 512×512 → SAM: 512/16=32 → /2/2 = **8×8 = 64**
- **Small (100 tokens)**: resize input to 640×640 → SAM: 640/16=40 → /2/2 = **10×10 = 100**

But here's the catch: if you change SAM's input resolution, its **output spatial grid no longer matches CLIP's native 16×16 config**. That's exactly why `get_abs_pos` (for CLIP) and `get_abs_pos_sam` (for SAM) exist in the file — they do **bicubic interpolation of the positional embeddings** to adapt a model pretrained/configured at one grid size to a different token count at inference time:

```python
if src_size != tgt_size:
    ...
    new_pos_embed = F.interpolate(old_pos_embed, size=(tgt_size, tgt_size), mode='bicubic', ...)
```

This is the mechanism the paper calls **"dynamic interpolation of positional encodings"** in Section 3.2.2 — it's what lets a single trained DeepEncoder support Tiny/Small/Base/Large/Gundam modes without retraining separate models for each resolution.

## To test 64 or 100 tokens

Just change the resize target in your preprocessing:
```python
transforms.Resize((512, 512))   # -> 64 tokens (Tiny)
# or
transforms.Resize((640, 640))   # -> 100 tokens (Small)
```
and update your shape assertions accordingly (e.g. `(1, 65, 1024)` for Tiny, `(1, 101, 1024)` for Small). The `get_abs_pos`/`get_abs_pos_sam` interpolation paths will kick in automatically since `src_size != tgt_size` in those cases — that's the actual code path this smoke test isn't currently exercising, and it'd be a good addition to check that the interpolation logic works end-to-end.

This file is the actual PyTorch implementation of DeepSeek-OCR's **DeepEncoder** — the vision encoder architecture described in the paper. It has three main sections:

## 1. `MlpProjector` class
A flexible projector module supporting many connector types between vision features and the LLM embedding space (`identity`, `linear`, `mlp_gelu`, `downsample_mlp_gelu`, `low_high_hybrid_split_mlp_gelu`, etc.), including:
- Token pooling (2×2 patch merging via unfold)
- Downsampling with padding logic for non-square token grids
- A static `get_flops_per_sample` method for FLOP accounting

This is generic infrastructure reused across DeepSeek's VLM family (not OCR-specific).

## 2. CLIP section (`build_clip_l`)
Implements the **CLIP-large** component of DeepEncoder:
- `CLIPVisionEmbeddings` — patch embedding + class token + interpolatable position embeddings (`get_abs_pos` does bicubic interpolation of pos-embeddings when resolution changes, which is exactly how DeepEncoder supports multiple resolutions)
- `NoTPAttention` / `NoTPFeedForward` / `NoTPTransformerBlock` / `NoTPTransformer` — a non-tensor-parallel transformer stack using `scaled_dot_product_attention` and `quick_gelu`
- `VitModel` — wraps embeddings + transformer, accepts pre-computed `patch_embeds` (i.e., it's designed to take input from the SAM/compressor stage rather than raw pixels — matching the paper's note that "we remove the first patch embedding layer")
- `vit_model_cfg` — the actual config: **24 layers, hidden_size 1024, 16 heads**, image_size 224, patch_size 14 — standard CLIP-Large (ViT-L/14) dimensions
- `build_clip_l()` — factory function instantiating this as the second (global attention) stage of DeepEncoder

## 3. SAM section (`build_sam_vit_b`, `_build_sam`)
Implements the **SAM-base (ViTDet)** component:
- `ImageEncoderViT` — the ViTDet backbone with absolute position embeddings, a stack of `Block`s, and a "neck" that projects to 256 channels, followed by `net_2`/`net_3` convolutions that progressively increase channels (256→512→1024) — this is the **16× convolutional downsampler/compressor** bridging SAM to CLIP described in the paper
- `Block` — transformer block supporting **window attention** (via `window_partition`/`window_unpartition`) or global attention depending on `window_size`, matching the paper's description of window attention dominating this stage
- `Attention` — multi-head attention with optional decomposed relative position embeddings (`add_decomposed_rel_pos`, `get_rel_pos`) — standard ViTDet/SAM design
- `PatchEmbed` — conv-based patch embedding
- `_build_sam` config: **depth 12, embed_dim 768, 12 heads, window_size 14, global_attn_indexes [2,5,8,11]**, image_size 1024, patch_size 16 — this is SAM's **ViT-B** encoder, confirming the paper's "SAM-base"
- `build_sam_vit_b` / `build_sam_fast_vit_b` — factory functions, with the fast variant using `torch.compile`

**In short:** this file contains the two encoder backbones (SAM-ViT-B for windowed local perception + CLIP-ViT-L for global attention) plus the convolutional compressor between them, and the projector module that would connect DeepEncoder's output tokens to the DeepSeek-3B-MoE decoder — i.e., the concrete code behind Figure 3 in the paper.

The paper's rationale for including SAM-base as the first stage (rather than just using CLIP alone) comes down to solving the high-resolution/low-activation-memory problem that plagues other VLM encoder designs (Section 2.1 and 3.2.1):

**1. Window attention keeps activation memory manageable at high resolution**
SAM-base (ViTDet architecture) is dominated by **window attention** rather than dense global attention. This means it can process high-resolution input (e.g., a 1024×1024 image → 4096 patch tokens at patch-size 16) without the massive activation memory blow-up that dense global attention would cause at that same resolution. The paper notes that since "the first half of encoder is dominated by window attention and only 80M, the activation is acceptable."

**2. It solves a problem CLIP alone can't**
If you fed a high-resolution image straight into a global-attention model like CLIP, you'd get the same problem Qwen2-VL's NaViT-style encoder has: "massive activation memory consumption that can cause GPU memory overflow" (Section 2.1). SAM's window attention acts as a buffer — it lets the model see fine-grained detail at high resolution cheaply, *before* the expensive global attention stage ever sees the data.

**3. It pairs naturally with the compressor to cut token count before the expensive stage**
SAM's output is downsampled 16× by the convolutional compressor (4096 → 256 tokens) *before* being passed to CLIP's global attention. So SAM handles the "look at everything locally" work at low cost, while the compressor ensures only a small, manageable number of tokens ever reach the computationally expensive dense-attention stage.

**4. Leverages pretraining gains**
SAM-base is a proven, pretrained architecture (originally from Segment Anything) — reusing it lets DeepEncoder benefit from existing perceptual pretraining rather than training a window-attention encoder from scratch.

In short: SAM handles **perception** (fine local detail at high resolution, cheaply), CLIP handles **knowledge** (global semantic understanding), and the compressor bridges them — this division of labor is exactly what lets DeepEncoder hit all five of the design goals listed in 3.2 (high resolution, low activation, few vision tokens, multi-resolution support, moderate parameters) that no existing single encoder satisfied.

DeepSeek-OCR's DeepEncoder uses **CLIP-large** (~300M parameters) as the component responsible for visual knowledge extraction via dense global attention.

A few relevant details from the paper:
- The DeepEncoder pairs a **SAM-base** (80M, ViTDet, patch-size 16) for window-attention-based perception with **CLIP-large** (300M) for global-attention-based knowledge extraction, connected in series via a 16× convolutional compressor.
- For the CLIP component, they **remove the first patch embedding layer**, since its input is no longer raw images but the downsampled token sequence coming from the SAM+compressor pipeline.
- Together, SAM-base + CLIP-large make up the ~380M-parameter DeepEncoder, which then feeds into the DeepSeek3B-MoE (570M activated params) decoder.

