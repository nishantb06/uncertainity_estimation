/mnt/data/checkpoints/checkpoints_406M_baseline_val_full_sam_clip_freeze_2/checkpoint_406M_baseline_val_full_sam_clip_freeze_2_step0000250_loss0.2144.ckpt

uv run python src/score_uncertainty.py \
  --checkpoint /mnt/data/checkpoints/checkpoints_406M_baseline_val_full_sam_clip_freeze_2/checkpoint_406M_baseline_val_full_sam_clip_freeze_2_step0000250_loss0.2144.ckpt \
  --batch-size 8 \
  --out-dir /mnt/data/uncertainity_estimation/uncertainty_scores_2


  uv run python src/score_ghost_uncertainty.py \
  --checkpoint /mnt/data/checkpoints/checkpoints_406M_baseline_val_full_sam_clip_freeze_2/checkpoint_406M_baseline_val_full_sam_clip_freeze_2_step0000250_loss0.2144.ckpt \
  --batch-size 4 \
  --out-dir /mnt/data/uncertainity_estimation/uncertainty_scores_3