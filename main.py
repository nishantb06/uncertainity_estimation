#!/usr/bin/env python3
"""YAML-driven entry point for DeepEncoder DocILE classification experiments."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
DOCILE = os.path.join(ROOT, "docile")
for path in (SRC, DOCILE):
    if path not in sys.path:
        sys.path.insert(0, path)

from classifier import build_document_classifier  # noqa: E402
from classes import ALLOWED_CLASSES  # noqa: E402
from dataloaders import build_train_eval_loaders  # noqa: E402
from training import (  # noqa: E402
    DocumentClassifierLightning,
    flatten_config,
    make_run_id,
    resolve_run_dirs,
    snapshot_config,
    vision_tokens_for_image_size,
    write_run_meta,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    for section in ("run", "data", "model", "training"):
        if section not in cfg:
            raise ValueError(f"Config missing required section '{section}': {path}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepEncoder classifier from YAML")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT, "configs", "baseline_val.yaml"),
        help="Path to experiment YAML",
    )
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    alias = run_cfg.get("alias")
    if not alias:
        raise ValueError("run.alias is required")

    pl.seed_everything(int(train_cfg.get("seed", 42)), workers=True)
    torch.set_float32_matmul_precision("high")

    batch_size = int(data_cfg["batch_size"])
    effective_batch_size = int(train_cfg["effective_batch_size"])
    if effective_batch_size % batch_size != 0:
        raise ValueError(
            f"effective_batch_size ({effective_batch_size}) must be divisible by "
            f"batch_size ({batch_size})"
        )

    class_list = list(data_cfg.get("class_list") or ALLOWED_CLASSES)
    model_cfg = {**model_cfg, "num_classes": int(model_cfg.get("num_classes", len(class_list)))}
    image_size = int(data_cfg.get("image_size", 1024))
    vision_tokens = vision_tokens_for_image_size(image_size)

    model = build_document_classifier(model_cfg)
    param_counts = model.count_parameters()
    run_id = make_run_id(param_counts["total"], alias)

    dirs = resolve_run_dirs(
        run_id,
        logs_root=train_cfg.get("logs_root", "/mnt/data/logs"),
        checkpoints_root=train_cfg.get("checkpoints_root", "/mnt/data/checkpoints"),
        replace_existing_logs=bool(train_cfg.get("replace_existing_logs", False)),
    )
    log_dir = dirs["log_dir"]
    checkpoint_dir = dirs["checkpoint_dir"]

    snapshot_config(config_path, log_dir, checkpoint_dir)
    meta = {
        "run_id": run_id,
        "alias": alias,
        "config_path": config_path,
        "train_mode": data_cfg["train_mode"],
        "sampled_documents_csv": data_cfg.get("sampled_documents_csv"),
        "N_total": param_counts["total"],
        "N_trainable": param_counts["trainable"],
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "image_size": image_size,
        "vision_tokens_per_image": vision_tokens,
        "max_steps": int(train_cfg["max_steps"]),
        "class_list": class_list,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "train_from_zero": model_cfg.get("train_from_zero"),
        "freeze": model_cfg.get("freeze"),
    }
    write_run_meta(checkpoint_dir, meta)

    print(
        f"run_id={run_id}  "
        f"N_total={param_counts['total']:,}  "
        f"N_trainable={param_counts['trainable']:,}"
    )
    print(f"logs -> {log_dir}")
    print(f"checkpoints -> {checkpoint_dir}")

    train_loader, eval_loader = build_train_eval_loaders(data_cfg)

    hparams = flatten_config(cfg)
    hparams.update(
        {
            "run_id": run_id,
            "N_total": param_counts["total"],
            "N_trainable": param_counts["trainable"],
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            "vision_tokens_per_image": vision_tokens,
        }
    )

    lit = DocumentClassifierLightning(
        model,
        lr=float(train_cfg["max_lr"]),
        warmup_steps=int(train_cfg["warmup_steps"]),
        max_steps=int(train_cfg["max_steps"]),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        class_names=class_list,
        hparams_to_log=hparams,
        vision_tokens_per_image=vision_tokens,
    )

    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="",
        version="",
        default_hp_metric=False,
    )

    ckpt_filename = (
        f"checkpoint_{run_id}_step{{step:07d}}_loss{{train_loss:.4f}}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=ckpt_filename,
        verbose=True,
        every_n_train_steps=int(train_cfg.get("save_checkpoints_every_n_steps", 500)),
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    progress_bar = RichProgressBar(
        refresh_rate=1,
        leave=False,
        theme=RichProgressBarTheme(
            description="",
            progress_bar="#6206E0",
            progress_bar_finished="#6206E0",
            progress_bar_pulse="#6206E0",
            batch_progress="",
            time="dim",
            processing_speed="dim underline",
            metrics="italic",
            metrics_text_delimiter=" ",
            metrics_format=".3f",
        ),
    )

    resume_path = os.path.join(checkpoint_dir, "last.ckpt")
    ckpt_path = resume_path if os.path.exists(resume_path) else None
    if ckpt_path is not None:
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        print("Starting training from scratch")

    val_check_interval = int(train_cfg.get("val_every_n_steps", 100))

    trainer = pl.Trainer(
        max_steps=int(train_cfg["max_steps"]),
        accelerator=device,
        devices=1,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            progress_bar,
            checkpoint_callback,
        ],
        precision=train_cfg.get("precision", "bf16-mixed"),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 10)),
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=tensorboard_logger,
        accumulate_grad_batches=effective_batch_size // batch_size,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )

    if bool(train_cfg.get("compile", False)):
        lit.enable_compile(mode="default")

    trainer.fit(
        lit,
        train_dataloaders=train_loader,
        val_dataloaders=eval_loader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
