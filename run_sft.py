#!/usr/bin/env python3
"""SFT training for AutoVLA on PAI dataset — no navsim/nuplan dependency.

Usage:
  AUTOVLA_REPO_PATH=/path/to/AutoVLA python run_sft.py --config config/pai_sft.yaml
"""
import sys
import os
import argparse
import pickle
from pathlib import Path

_AUTOVLA_REPO = os.environ.get("AUTOVLA_REPO_PATH", "/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA")
sys.path.insert(0, _AUTOVLA_REPO)

import yaml
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import datetime

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pai_sft_dataset import PAISFTDataset, PAIDataCollator


class LightSFTAutoVLA(pl.LightningModule):
    """Lightweight SFT model: Qwen2.5-VL + action tokens, no navsim/nuplan."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config["model"]

        # Load Qwen model
        model_path = model_cfg["pretrained_model_path"]
        print(f"Loading Qwen2.5-VL from {model_path}...")
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        # Add action tokens (matching AutoVLA's ActionTokenizer)
        codebook_path = model_cfg["codebook_cache_path"]
        with open(codebook_path, "rb") as f:
            codebook = pickle.load(f)
        n_bins = len(codebook["token_all"]["veh"])
        action_tokens = [f"<action_{i}>" for i in range(n_bins)]
        added = self.processor.tokenizer.add_tokens(action_tokens, special_tokens=False)
        print(f"Added {added} action tokens (n_bins={n_bins})")

        # Resize embeddings
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))

        # Enable gradient checkpointing
        self.vlm.model.gradient_checkpointing_enable()

        # Training config
        self.lr = config["training"]["learning_rate"]
        self.ignore_index = model_cfg["tokens"]["ignore_index"]

    def forward(self, batch):
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "pixel_values_videos" in batch:
            kwargs["pixel_values_videos"] = batch["pixel_values_videos"]
        if "video_grid_thw" in batch:
            kwargs["video_grid_thw"] = batch["video_grid_thw"]
        if "pixel_values" in batch:
            kwargs["pixel_values"] = batch["pixel_values"]
        if "image_grid_thw" in batch:
            kwargs["image_grid_thw"] = batch["image_grid_thw"]
        return self.vlm(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=self.ignore_index,
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=self.ignore_index,
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup = self.config["training"].get("lr_warmup_step", 500)
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.config["training"].get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda step: min(1.0, step / max(1, warmup)),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


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

    model_cfg = config["model"]
    training_cfg = config["training"]

    processor = AutoProcessor.from_pretrained(model_cfg["pretrained_model_path"], use_fast=True)

    train_dataset = PAISFTDataset(
        pai_data_dir=config["data"]["pai_data_dir"],
        model_config=model_cfg,
        processor=processor,
        using_cot=model_cfg.get("use_cot", False),
        max_samples=training_cfg.get("train_sample_size"),
    )
    val_dataset = PAISFTDataset(
        pai_data_dir=config["data"]["pai_data_dir"],
        model_config=model_cfg,
        processor=processor,
        using_cot=model_cfg.get("use_cot", False),
        max_samples=training_cfg.get("val_sample_size", 50),
    )
    collator = PAIDataCollator(processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        collate_fn=collator,
        num_workers=training_cfg["num_workers"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=collator,
        num_workers=2,
        shuffle=False,
    )

    model = LightSFTAutoVLA(config)

    # Load SFT checkpoint if specified
    ckpt_path = model_cfg.get("sft_model_path")
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading SFT checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Remap keys: model.visual.* → visual.*, model.language_model.* → model.*
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("model.visual."):
                new_k = k.replace("model.visual.", "visual.", 1)
            elif k.startswith("model.language_model."):
                new_k = k.replace("model.language_model.", "model.", 1)
            else:
                new_k = k
            new_state[new_k] = v
        # Filter to matching keys
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in new_state.items() if k in model_keys}
        skipped = len(new_state) - len(filtered)
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(new_state)} keys (skipped {skipped} non-matching)")

    save_dir = f"runs/sft/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=training_cfg["epochs"],
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 4),
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                dirpath=save_dir,
                filename="epoch={epoch}-val={val_loss:.4f}",
                auto_insert_metric_name=False,
                save_weights_only=True,
                every_n_epochs=1,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=CSVLogger(save_dir=save_dir),
        enable_model_summary=True,
    )

    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_path = os.path.join(save_dir, "final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"\nSFT complete! Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
