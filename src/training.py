"""Lightning training module and experiment path helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import lightning.pytorch as pl
from classifier import DocumentClassifier
from classes import ALLOWED_CLASSES

# DeepEncoder: spatial tokens = (H/64) * (W/64) for square inputs.
VISION_TOKENS_BY_SIZE: dict[int, int] = {
    512: 64,
    640: 100,
    1024: 256,
}


def vision_tokens_for_image_size(image_size: int) -> int:
    if image_size in VISION_TOKENS_BY_SIZE:
        return VISION_TOKENS_BY_SIZE[image_size]
    grid = max(1, image_size // 64)
    return grid * grid


def format_param_tag(n_params: int) -> str:
    """Format a parameter count as a short tag, e.g. 50_123_456 -> '50M'."""
    if n_params >= 1_000_000_000:
        billions = n_params / 1_000_000_000
        if abs(billions - round(billions)) < 0.05:
            return f"{int(round(billions))}B"
        return f"{billions:.1f}B".replace(".0B", "B")
    millions = n_params / 1_000_000
    if millions >= 1:
        if abs(millions - round(millions)) < 0.5:
            return f"{int(round(millions))}M"
        return f"{millions:.1f}M".replace(".0M", "M")
    thousands = max(1, int(round(n_params / 1_000)))
    return f"{thousands}K"


def make_run_id(n_params: int, alias: str) -> str:
    tag = format_param_tag(n_params)
    alias_clean = alias.strip().replace(" ", "-")
    return f"{tag}_{alias_clean}"


def resolve_run_dirs(
    run_id: str,
    *,
    logs_root: str = "/mnt/data/logs",
    checkpoints_root: str = "/mnt/data/checkpoints",
    replace_existing_logs: bool = False,
) -> dict[str, str]:
    os.makedirs(logs_root, exist_ok=True)
    os.makedirs(checkpoints_root, exist_ok=True)

    log_dir = os.path.join(logs_root, f"logs_{run_id}")
    checkpoint_dir = os.path.join(checkpoints_root, f"checkpoints_{run_id}")

    if replace_existing_logs and os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return {"log_dir": log_dir, "checkpoint_dir": checkpoint_dir, "run_id": run_id}


def write_run_meta(checkpoint_dir: str, meta: dict[str, Any]) -> str:
    path = os.path.join(checkpoint_dir, "run_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return path


def snapshot_config(config_path: str, *dest_dirs: str) -> None:
    for dest_dir in dest_dirs:
        dest = os.path.join(dest_dir, "config.yaml")
        shutil.copy2(config_path, dest)


def flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        full = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(flatten_config(value, full))
        elif isinstance(value, (bool, int, float, str)) or value is None:
            flat[full] = "null" if value is None else value
        else:
            flat[full] = str(value)
    return flat


class DocumentClassifierLightning(pl.LightningModule):
    def __init__(
        self,
        model: DocumentClassifier,
        *,
        lr: float,
        warmup_steps: int,
        max_steps: int,
        grad_clip: float = 1.0,
        class_names: list[str] | None = None,
        hparams_to_log: dict | None = None,
        vision_tokens_per_image: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "hparams_to_log"])
        self.model = model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.criterion = nn.CrossEntropyLoss()
        self.class_names = class_names or list(ALLOWED_CLASSES)
        self.hparams_to_log = hparams_to_log or {}
        self.vision_tokens_per_image = int(vision_tokens_per_image)
        self._compiled_model = None

        # Accumulators for epoch-end metrics
        self._val_correct = 0
        self._val_total = 0
        self._val_correct_per_class: dict[int, int] = defaultdict(int)
        self._val_total_per_class: dict[int, int] = defaultdict(int)

        # Throughput timing
        self._batch_start_time: float | None = None
        self._opt_step_start_time: float | None = None
        self._images_in_opt_step = 0

    def enable_compile(self, mode: str = "default") -> None:
        self._compiled_model = torch.compile(self.model, mode=mode)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._compiled_model is not None:
            return self._compiled_model(pixel_values)
        return self.model(pixel_values)

    def _sync_cuda(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _accumulate_grad_batches(self) -> int:
        accumulate = self.trainer.accumulate_grad_batches
        if isinstance(accumulate, dict):
            return int(accumulate.get(0, 1))
        return int(accumulate)

    def on_fit_start(self) -> None:
        if self.hparams_to_log and self.logger is not None:
            self.logger.log_hyperparams(self.hparams_to_log)
            for key in (
                "N_total",
                "N_trainable",
                "batch_size",
                "effective_batch_size",
                "vision_tokens_per_image",
            ):
                if key in self.hparams_to_log:
                    self.logger.log_metrics(
                        {f"params/{key}": float(self.hparams_to_log[key])},
                        step=0,
                    )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self._opt_step_start_time is None:
            self._sync_cuda()
            self._opt_step_start_time = time.perf_counter()
        self._sync_cuda()
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._sync_cuda()
        if self._batch_start_time is None:
            return

        batch_time = max(time.perf_counter() - self._batch_start_time, 1e-9)
        n_images = int(batch["pixel_values"].shape[0])
        micro_ips = n_images / batch_time
        micro_tps = micro_ips * self.vision_tokens_per_image

        self.log(
            "throughput/images_per_sec",
            micro_ips,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=n_images,
        )
        self.log(
            "throughput/tokens_per_sec",
            micro_tps,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=n_images,
        )

        self._images_in_opt_step += n_images

        if (batch_idx + 1) % self._accumulate_grad_batches() != 0:
            return
        if self._opt_step_start_time is None:
            return

        opt_step_time = max(time.perf_counter() - self._opt_step_start_time, 1e-9)
        train_ips = self._images_in_opt_step / opt_step_time
        train_tps = train_ips * self.vision_tokens_per_image
        self.log(
            "throughput/train_images_per_sec",
            train_ips,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self._images_in_opt_step,
        )
        self.log(
            "throughput/train_tokens_per_sec",
            train_tps,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self._images_in_opt_step,
        )
        self._images_in_opt_step = 0
        self._opt_step_start_time = None

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["pixel_values"])
        loss = self.criterion(logits, batch["labels"])
        preds = logits.argmax(dim=-1)
        acc = (preds == batch["labels"]).float().mean()
        bs = int(batch["labels"].shape[0])
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=bs,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_correct = 0
        self._val_total = 0
        self._val_correct_per_class = defaultdict(int)
        self._val_total_per_class = defaultdict(int)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pixel_values"])
        loss = self.criterion(logits, batch["labels"])
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        correct = preds == labels
        bs = int(labels.shape[0])
        self._val_correct += int(correct.sum().item())
        self._val_total += int(labels.numel())
        for p, y, ok in zip(preds.tolist(), labels.tolist(), correct.tolist()):
            self._val_total_per_class[y] += 1
            if ok:
                self._val_correct_per_class[y] += 1
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )

    def on_validation_epoch_end(self) -> None:
        acc = self._val_correct / max(1, self._val_total)
        self.log("val_acc", acc, prog_bar=True, sync_dist=False, batch_size=self._val_total)
        for idx, name in enumerate(self.class_names):
            total = self._val_total_per_class.get(idx, 0)
            correct = self._val_correct_per_class.get(idx, 0)
            class_acc = correct / total if total else 0.0
            self.log(
                f"val_acc_per_class/{name}",
                class_acc,
                sync_dist=False,
                batch_size=max(1, total),
            )

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.grad_clip and self.grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.log(
                "grad_norm",
                norm,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=1,
            )


    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr)

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return (current_step + 1) / max(1, self.warmup_steps)
            if current_step > self.max_steps:
                return 0.1
            decay_ratio = (current_step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return 0.1 + coeff * 0.9

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
