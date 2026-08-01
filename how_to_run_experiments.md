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

