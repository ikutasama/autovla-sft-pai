#!/usr/bin/env python3
"""SFT training script for AutoVLA on PAI dataset.

Usage:
  AUTOVLA_REPO_PATH=/path/to/AutoVLA python run_sft.py --config config/pai_sft.yaml
"""
import sys
import os
import argparse
from pathlib import Path

# Add AutoVLA repo to path
_AUTOVLA_REPO = os.environ.get("AUTOVLA_REPO_PATH", "/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA")
sys.path.insert(0, _AUTOVLA_REPO)

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.utils.data import DataLoader
import functools
import datetime

from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

from pai_sft_dataset import PAISFTDataset, PAIDataCollator


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    config = load_config(args.config)

    # Model config
    model_config = config["model"]
    training_config = config["training"]

    # Processor
    processor = AutoProcessor.from_pretrained(
        model_config["pretrained_model_path"], use_fast=True
    )

    # Datasets
    train_dataset = PAISFTDataset(
        pai_data_dir=config["data"]["pai_data_dir"],
        model_config=model_config,
        processor=processor,
        using_cot=model_config.get("use_cot", False),
        max_samples=training_config.get("train_sample_size"),
    )

    val_dataset = PAISFTDataset(
        pai_data_dir=config["data"]["pai_data_dir"],
        model_config=model_config,
        processor=processor,
        using_cot=model_config.get("use_cot", False),
        max_samples=training_config.get("val_sample_size", 50),
    )

    data_collator = PAIDataCollator(processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        collate_fn=data_collator,
        num_workers=training_config["num_workers"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=data_collator,
        num_workers=2,
        shuffle=False,
    )

    # Model: load AutoVLA SFT model
    from models.autovla import SFTAutoVLA
    model = SFTAutoVLA(config)

    # Load existing SFT checkpoint if specified
    ckpt_path = model_config.get("sft_model_path")
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading SFT checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        print(f"Loaded SFT checkpoint (missing keys will be fine for resized embeddings)")

    # Enable gradient checkpointing
    model.autovla.vlm.model.gradient_checkpointing_enable()

    # Training
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2_5_VLDecoderLayer},
    )

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"runs/sft/{current_date}"

    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=training_config["epochs"],
        accelerator="gpu",
        devices="auto",
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 4),
        strategy=FSDPStrategy(
            auto_wrap_policy=wrap_policy,
            cpu_offload=False,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            sharding_strategy="FULL_SHARD",
            backward_prefetch="BACKWARD_PRE",
            state_dict_type="full",
            limit_all_gathers=True,
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                dirpath=save_dir,
                filename="epoch={epoch}-loss={val_loss:.4f}",
                auto_insert_metric_name=False,
                save_weights_only=True,
                every_n_epochs=1,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        logger=CSVLogger(save_dir=save_dir),
        enable_model_summary=True,
    )

    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Export final checkpoint
    final_path = os.path.join(save_dir, "final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"\nSFT complete! Final checkpoint: {final_path}")
    print(f"Use this checkpoint for GRPO: set sft_model_path in alpagym config")


if __name__ == "__main__":
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    main()
