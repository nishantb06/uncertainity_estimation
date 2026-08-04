# DeepEncoder DocILE classification experiments

Training stack mirrors `scaling-laws`: YAML configs → `main.py` → Lightning + TensorBoard.
Artifacts land under `/mnt/data/logs` and `/mnt/data/checkpoints`.

## One-time setup

```bash
cd /mnt/data/uncertainity_estimation
uv sync
# System poppler required for page_image():
#   sudo dnf install -y poppler-utils

# Create stratified 180-doc holdout (val classes only; drops debit_note/utility_bill)
uv run python src/make_holdout_split.py
```

This writes:

- `splits/holdout_180.json` — fixed eval set (never used for training)
- `splits/train_pool.json` — eligible train minus holdout (~4982)
- `splits/dropped_excluded_classes.json` — audit of dropped train docs
- `splits/holdout_180_meta.json` — class counts

## Train

```bash
# Ceiling: (train_pool ∪ val) → evaluate on holdout
uv run python main.py --config configs/ceiling.yaml

# Baseline: val only
uv run python main.py --config configs/baseline_val.yaml

# Later: val + 1500 sampled from train_pool
# Fill splits/sampled_1500_placeholder.csv (header doc_id, one id per row), then:
uv run python main.py --config configs/selected_placeholder.yaml
```

Run directories use `run_id = {param_tag}_{alias}`, e.g. `logs_380M_ceiling_full/`, `checkpoints_380M_entropy_estimation/`.

Resume is automatic from `last.ckpt` in the run’s checkpoint directory.

## Config knobs

- `run.alias` — appears in log/checkpoint names
- `data.train_mode` — `ceiling` | `val` | `val_plus_sampled`
- `data.sampled_documents_csv` — `null` or path to CSV of pool ids
- `model.train_from_zero.SAM/CLIP` — skip loading that tower’s pretrained weights
- `model.freeze.SAM/CLIP` — freeze when loaded (ignored / always trainable if from-zero)

## TensorBoard

```bash
uv run tensorboard --logdir /mnt/data/logs --bind_all --port 6006
```

Compare `val_acc` / `val_loss` across `logs_*` runs.

Throughput (CUDA-synchronized wall time):

- `throughput/images_per_sec` — micro-batch images/s (also on progress bar)
- `throughput/train_images_per_sec` — images/s over a full optimizer step (incl. grad accumulation)
- `throughput/tokens_per_sec` / `throughput/train_tokens_per_sec` — same × vision tokens/image (`image_size/64`²)

## Uncertainty scoring (entropy / JSD)

Score `splits/train_pool.json` with a trained classifier at resolutions 512, 640, 1024 (natural log):

```bash
uv run python src/score_uncertainty.py \
  --checkpoint /mnt/data/checkpoints/checkpoints_406M_baseline_val_full_sam_clip_freeze_2/last.ckpt \
  --batch-size 8 \
  --out-dir /mnt/data/uncertainity_estimation/uncertainty_scores
```

Uses `config.yaml` next to the checkpoint by default. Outputs under
`uncertainty_scores/<checkpoints_run_folder>/`:

| File | Contents |
|------|----------|
| `{stem}_entropy_information.csv` | All pool docs: `entropy_1024/640/512`, `entropy_mean_logits`, `jsd_1024_512`, `jsd_1024_640`, `jsd_512_640`, plus `pred_*` |
| `{stem}_top1500_by_entropy_mean_logits.csv` | Top-K ids by `entropy_mean_logits` (for `sampled_documents_csv`) |
| `{stem}_hist_*.png` | Histograms per metric |
| `{stem}_summary.json` | Means/stds and paths |

Optional flags: `--doc-ids-json`, `--top-k`, `--rank-by` (any metric column), `--config`, `--no-amp`.

## Ghost-gradient uncertainty (multi-resolution alignment)

Scores the same `train_pool` by comparing **head** weight-gradient directions across
resolutions 1024 / 640 / 512 using the ghost trick (no full per-sample grad matrices).

```bash
uv run python src/score_ghost_uncertainty.py \
  --checkpoint /mnt/data/checkpoints/checkpoints_406M_baseline_val_full_sam_clip_freeze_2/last.ckpt \
  --batch-size 4 \
  --out-dir /mnt/data/uncertainity_estimation/uncertainty_scores
```

| File | Contents |
|------|----------|
| `{stem}_ghost_information.csv` | `ghost_cos_*` pairs, `ghost_avg_cos`, `ghost_uncertainty` (= `1 - avg_cos`) |
| `{stem}_top1500_by_ghost_uncertainty.csv` | Top-K most uncertain ids |
| `{stem}_hist_ghost_*.png` | Histograms |
| `{stem}_ghost_summary.json` | Means/stds and paths |

Encoder runs under `no_grad`; only MLP head Linears are hooked. AdamW `exp_avg_sq`
from the ckpt is used as a row/col preconditioner when present (`--no-precond` to disable).

