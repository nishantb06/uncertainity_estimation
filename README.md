# Uncertainty estimation

Select which unlabeled DocILE documents to annotate so a document classifier improves most for a fixed labeling budget.

Full writeup: [REPORT.md](REPORT.md) · PDF: [report.pdf](report.pdf) · Repo: https://github.com/nishantb06/uncertainity_estimation

## Protocol

| Split | Role |
|---|---|
| Original val (500) | Labeled seed |
| Original train (5180) | Unlabeled pool |
| Holdout (180) | Fixed eval (stratified; never used for training) |
| Train pool (~4982) | Selectable docs after holdout + dropping `debit_note` / `utility_bill` |

Three runs, all evaluated on holdout:

1. **Baseline** — seed only (500)
2. **Selected** — seed + 1500 chosen from the pool (labels unused at selection time)
3. **Ceiling** — full pool ∪ seed (upper bound)

## Model

DeepSeek OCR vision encoder (SAM → downsample → CLIP), SAM/CLIP frozen, custom MLP on CLIP’s CLS token for 7-way document-type classification (page 0, train at 640). Tax invoice and order dominate (~68% / ~25%); report curves focus on those two.

## Selection

Score the pool with the baseline checkpoint at resolutions `{512, 640, 1024}`. Main rule: **average entropy** — entropy of the mean logits across resolutions (`entropy_mean_logits`), take top 1500.

Also implemented: per-resolution entropy, Jensen–Shannon divergence across resolutions, and ghost-gradient disagreement of MLP-head gradients.

## Results (summary)

Average-entropy 500 + 1500 beats baseline on holdout loss and tax-invoice / order accuracy, but does not beat the ceiling. See [REPORT.md](REPORT.md) for equations, plots, and limitations.

## Quick start

```bash
uv sync
# needs poppler for page_image()
uv run python src/make_holdout_split.py

uv run python main.py --config configs/baseline_val.yaml
uv run python main.py --config configs/ceiling.yaml

# score pool → top-1500 CSV, then train selected
uv run python src/score_uncertainty.py \
  --checkpoint /path/to/baseline/last.ckpt \
  --batch-size 8 \
  --out-dir uncertainty_scores
uv run python main.py --config configs/selected_placeholder.yaml
```

Training, TensorBoard, and ghost scoring details: [how_to_run_experiments.md](how_to_run_experiments.md)

## Layout

```
configs/          # baseline / ceiling / selected YAMLs
src/              # model, training, entropy & ghost scorers
splits/           # holdout, train_pool, sampled CSVs
images/           # report figures
REPORT.md         # experiment writeup
```
