#!/usr/bin/env python3
"""Single-GPU SFT for AutoVLA on the local NVIDIA PAI AV dataset.

Run this on the A100 server with alpagym's existing ``.venv/bin/python`` rather
than the ambient conda interpreter. See README.md for the exact command.
"""

import argparse
import datetime
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from pai_sft_dataset import PAIDataCollator, PAISFTDataset, SimplePAIInterface


torch.set_float32_matmul_precision("high")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_file_or_dir(path: str, description: str) -> Path:
    resolved = Path(os.path.expandvars(os.path.expanduser(path)))
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def configure_action_tokens(processor, model_cfg: dict) -> Tuple[int, Tuple[int, ...]]:
    """Install the AutoVLA action vocabulary on the one shared processor."""
    codebook_path = _require_file_or_dir(
        model_cfg["codebook_cache_path"], "Action codebook"
    )
    with codebook_path.open("rb") as f:
        codebook = pickle.load(f)

    try:
        n_bins = len(codebook["token_all"]["veh"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid AutoVLA codebook structure in {codebook_path}"
        ) from exc

    action_tokens = [f"<action_{i}>" for i in range(n_bins)]
    added = processor.tokenizer.add_tokens(action_tokens, special_tokens=False)
    action_ids = tuple(processor.tokenizer.convert_tokens_to_ids(action_tokens))
    expected_start = int(model_cfg["tokens"]["action_start_id"])
    expected_ids = tuple(range(expected_start, expected_start + n_bins))
    if action_ids != expected_ids:
        raise RuntimeError(
            "Action token IDs are incompatible with AutoVLA. "
            f"Expected [{expected_start}, {expected_start + n_bins - 1}], "
            f"got [{action_ids[0]}, {action_ids[-1]}]. Use the exact base "
            "Qwen2.5-VL tokenizer used by the PDMS checkpoint."
        )

    print(
        f"Action vocabulary ready: {n_bins} tokens, IDs "
        f"{action_ids[0]}..{action_ids[-1]} ({added} newly added)"
    )
    return n_bins, action_ids


def normalize_checkpoint_key(key: str) -> str:
    """Map published AutoVLA/Lightning checkpoint keys to LightSFTAutoVLA."""
    wrapper_prefixes = ("_forward_module.", "module.")
    model_prefixes = ("autovla.", "drivevla.")

    changed = True
    while changed:
        changed = False
        for prefix in wrapper_prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
        for prefix in model_prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True

    if key.startswith("vlm."):
        return key
    if key.startswith("model.visual."):
        return f"vlm.visual.{key[len('model.visual.') :]}"
    if key.startswith("model.language_model."):
        return f"vlm.model.{key[len('model.language_model.') :]}"
    if key.startswith(("model.", "visual.", "lm_head.")):
        return f"vlm.{key}"
    return key


def checkpoint_key_candidates(key: str) -> Tuple[str, ...]:
    """Return key layouts used across Transformers and AutoVLA revisions."""
    primary = normalize_checkpoint_key(key)
    candidates = [primary]
    if primary.startswith("vlm.visual."):
        candidates.append(f"vlm.model.visual.{primary[len('vlm.visual.') :]}")
    elif primary.startswith("vlm.model.visual."):
        candidates.append(f"vlm.visual.{primary[len('vlm.model.visual.') :]}")
    elif primary.startswith("vlm.model.language_model."):
        candidates.append(f"vlm.model.{primary[len('vlm.model.language_model.') :]}")
    elif primary.startswith("vlm.model."):
        candidates.append(f"vlm.model.language_model.{primary[len('vlm.model.') :]}")
    return tuple(dict.fromkeys(candidates))


def load_sft_checkpoint(model: torch.nn.Module, model_cfg: dict) -> None:
    checkpoint_path = _require_file_or_dir(
        model_cfg["sft_model_path"], "Starting SFT checkpoint"
    )
    print(f"Loading starting SFT checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"No state_dict found in checkpoint: {checkpoint_path}")

    model_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    unexpected = []
    shape_mismatches = []
    for source_key, value in state_dict.items():
        target_key = next(
            (
                candidate
                for candidate in checkpoint_key_candidates(source_key)
                if candidate in model_state
                and model_state[candidate].shape == value.shape
            ),
            None,
        )
        known_candidates = [
            candidate
            for candidate in checkpoint_key_candidates(source_key)
            if candidate in model_state
        ]
        if target_key is None and not known_candidates:
            unexpected.append(source_key)
        elif target_key is None:
            shape_target = known_candidates[0]
            shape_mismatches.append(
                (source_key, tuple(value.shape), tuple(model_state[shape_target].shape))
            )
        else:
            compatible[target_key] = value

    total_numel = sum(value.numel() for value in model_state.values())
    loaded_numel = sum(model_state[key].numel() for key in compatible)
    load_ratio = loaded_numel / max(1, total_numel)
    minimum_ratio = float(model_cfg.get("min_checkpoint_load_ratio", 0.95))
    if load_ratio < minimum_ratio:
        mismatch_preview = ", ".join(
            f"{key}: {source_shape}->{target_shape}"
            for key, source_shape, target_shape in shape_mismatches[:3]
        )
        raise RuntimeError(
            f"Checkpoint load coverage is only {load_ratio:.2%} "
            f"({len(compatible)}/{len(model_state)} model tensors); expected at "
            f"least {minimum_ratio:.0%}. Unexpected keys={len(unexpected)}, "
            f"shape mismatches={len(shape_mismatches)}. {mismatch_preview}"
        )

    missing, _ = model.load_state_dict(compatible, strict=False)
    print(
        f"Checkpoint loaded: {load_ratio:.2%} of model parameters, "
        f"{len(compatible)} tensors, {len(missing)} missing tensors, "
        f"{len(unexpected)} unrelated tensors"
    )

    del checkpoint, state_dict, compatible


class LightSFTAutoVLA(pl.LightningModule):
    """Qwen2.5-VL plus AutoVLA action tokens, without navsim/nuplan."""

    def __init__(self, config: dict, processor, n_action_tokens: int):
        super().__init__()
        self.config = config
        self.processor = processor
        self.n_action_tokens = n_action_tokens
        model_cfg = config["model"]

        model_path = _require_file_or_dir(
            model_cfg["pretrained_model_path"], "Qwen2.5-VL model"
        )
        print(f"Loading Qwen2.5-VL from {model_path}...")
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_cfg.get(
                "attn_implementation", "flash_attention_2"
            ),
        )
        self.vlm.resize_token_embeddings(len(processor.tokenizer))
        self.vlm.config.use_cache = False
        self.vlm.gradient_checkpointing_enable()

        vision_backbone = getattr(self.vlm, "visual", None)
        if vision_backbone is None:
            vision_backbone = getattr(getattr(self.vlm, "model", None), "visual", None)
        if vision_backbone is None:
            raise RuntimeError("Could not locate the Qwen2.5-VL vision backbone")

        if not model_cfg.get("train_vision_backbone", False):
            vision_backbone.requires_grad_(False)
            print("Vision backbone frozen")
        else:
            print("Vision backbone trainable")

        if not model_cfg.get("train_lm_backbone", True):
            language_backbone = self.vlm.model
            if hasattr(language_backbone, "language_model"):
                language_backbone = language_backbone.language_model
            language_backbone.requires_grad_(False)
            self.vlm.get_input_embeddings().weight.requires_grad_(True)
            output_embeddings = self.vlm.get_output_embeddings()
            if output_embeddings is not None:
                output_embeddings.weight.requires_grad_(True)
            print("Language backbone frozen; token embeddings remain trainable")

        self.lr = float(config["training"]["learning_rate"])
        self.action_start_id = int(model_cfg["tokens"]["action_start_id"])
        self.action_end_id = self.action_start_id + n_action_tokens

    def forward(self, batch):
        accepted_keys = (
            "input_ids",
            "attention_mask",
            "pixel_values_videos",
            "video_grid_thw",
            "pixel_values",
            "image_grid_thw",
            "labels",
        )
        return self.vlm(**{key: batch[key] for key in accepted_keys if key in batch})

    def _shared_step(self, batch, stage: str):
        outputs = self(batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {stage} loss: {loss}")

        shift_labels = batch["labels"][:, 1:]
        action_mask = (shift_labels >= self.action_start_id) & (
            shift_labels < self.action_end_id
        )
        if action_mask.any():
            predictions = outputs.logits[:, :-1].argmax(dim=-1)
            action_accuracy = (
                (predictions[action_mask] == shift_labels[action_mask]).float().mean()
            )
            self.log(
                f"{stage}_action_accuracy",
                action_accuracy,
                prog_bar=stage == "val",
                batch_size=batch["input_ids"].shape[0],
            )

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            batch_size=batch["input_ids"].shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        training_cfg = self.config["training"]
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable model parameters")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.lr,
            weight_decay=float(training_cfg.get("weight_decay", 0.01)),
            betas=(0.9, 0.999),
        )
        warmup = int(training_cfg.get("lr_warmup_step", 500))
        step_frequency = int(training_cfg.get("lr_step_frequency", 2000))
        gamma = float(training_cfg.get("lr_step_gamma", 0.98))

        def lr_scale(step: int) -> float:
            if warmup > 0 and step < warmup:
                return 0.05 + 0.95 * step / warmup
            decay_steps = max(0, step - warmup) // max(1, step_frequency)
            return max(0.01, gamma**decay_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def build_dataloaders(config: dict, processor, n_action_tokens: int):
    model_cfg = config["model"]
    training_cfg = config["training"]
    data_cfg = config["data"]
    interface = SimplePAIInterface(data_cfg["pai_data_dir"])

    common_dataset_args = {
        "dataset_interface": interface,
        "data_config": data_cfg,
        "model_config": model_cfg,
        "processor": processor,
    }
    train_dataset = PAISFTDataset(
        **common_dataset_args,
        split="train",
        max_samples=training_cfg.get("train_sample_size"),
    )
    val_dataset = PAISFTDataset(
        **common_dataset_args,
        split="val",
        max_samples=training_cfg.get("val_sample_size", 200),
    )
    overlap = set(train_dataset.clip_ids).intersection(val_dataset.clip_ids)
    if overlap:
        raise RuntimeError(
            f"Train/validation clip leakage detected: {len(overlap)} clips"
        )

    collator = PAIDataCollator(
        processor=processor,
        ignore_index=int(model_cfg["tokens"]["ignore_index"]),
        action_start_id=int(model_cfg["tokens"]["action_start_id"]),
        n_action_tokens=n_action_tokens,
    )
    loader_args = {
        "collate_fn": collator,
        "pin_memory": True,
        "persistent_workers": int(training_cfg["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(training_cfg["num_workers"]),
        shuffle=True,
        **loader_args,
    )
    val_workers = int(config.get("inference", {}).get("num_workers", 2))
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("inference", {}).get("batch_size", 1)),
        num_workers=val_workers,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=val_workers > 0,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def run_data_preflight(train_dataset, val_dataset, train_loader) -> None:
    preflight_batch_size = min(train_loader.batch_size, len(train_dataset))
    batch = train_loader.collate_fn(
        [train_dataset[index] for index in range(preflight_batch_size)]
    )
    action_start = train_loader.collate_fn.action_start_id
    action_end = action_start + train_loader.collate_fn.n_action_tokens
    action_count = int(
        ((batch["labels"] >= action_start) & (batch["labels"] < action_end)).sum()
    )
    trajectory = batch["gt_trajectory"][0]
    print("\nData preflight passed")
    print(f"  train clips: {len(train_dataset)}")
    print(f"  val clips:   {len(val_dataset)}")
    print(f"  batch action labels: {action_count}")
    print(f"  first waypoint: {trajectory[0].tolist()}")
    print(f"  final waypoint: {trajectory[-1].tolist()}")
    print(f"  input shape: {tuple(batch['input_ids'].shape)}")


def apply_smoke_overrides(config: dict) -> None:
    training_cfg = config["training"]
    training_cfg.update(
        {
            "epochs": 1,
            "train_sample_size": 8,
            "val_sample_size": 4,
            "accumulate_grad_batches": 1,
            "lr_warmup_step": 2,
        }
    )
    print("Smoke-test overrides enabled: 8 train clips, 4 val clips, 1 epoch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, split, video decoding and action labels without loading the model.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run an 8-sample, one-epoch overfit test before full training.",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    config = load_config(args.config)
    if args.smoke_test:
        apply_smoke_overrides(config)

    model_cfg = config["model"]
    model_path = _require_file_or_dir(
        model_cfg["pretrained_model_path"], "Qwen2.5-VL model"
    )
    _require_file_or_dir(model_cfg["sft_model_path"], "Starting SFT checkpoint")
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    n_action_tokens, _ = configure_action_tokens(processor, model_cfg)
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        config, processor, n_action_tokens
    )
    run_data_preflight(train_dataset, val_dataset, train_loader)
    if args.preflight_only:
        return

    model = LightSFTAutoVLA(config, processor, n_action_tokens)
    load_sft_checkpoint(model, model_cfg)

    save_dir = Path("runs/sft") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=False)
    processor.save_pretrained(save_dir / "processor")
    training_cfg = config["training"]
    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=int(training_cfg["epochs"]),
        accelerator="gpu",
        devices=int(training_cfg.get("devices", 1)),
        strategy=training_cfg.get("strategy", "auto"),
        precision=training_cfg.get("precision", "bf16-mixed"),
        accumulate_grad_batches=int(training_cfg.get("accumulate_grad_batches", 4)),
        gradient_clip_val=1.0,
    )
    ckpt_steps = int(training_cfg.get("checkpoint_every_n_steps", 0)) or None
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss" if not ckpt_steps else None,
            mode="min",
            save_top_k=3 if not ckpt_steps else -1,
            dirpath=save_dir,
            filename="step={step}-loss={train_loss:.4f}" if ckpt_steps else "epoch={epoch}-val={val_loss:.4f}",
            auto_insert_metric_name=False,
            save_weights_only=True,
            every_n_train_steps=ckpt_steps,
            every_n_epochs=None if ckpt_steps else 1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    val_interval = training_cfg.get("val_check_interval")
    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=int(training_cfg["epochs"]),
        val_check_interval=val_interval if val_interval else 1.0,
        check_val_every_n_epoch=None if val_interval else 1,
        accelerator="gpu",
        devices=int(training_cfg.get("devices", 1)),
        strategy=training_cfg.get("strategy", "auto"),
        precision=training_cfg.get("precision", "bf16-mixed"),
        accumulate_grad_batches=int(training_cfg.get("accumulate_grad_batches", 4)),
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        enable_model_summary=True,
    )

    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)

    final_path = save_dir / "final.ckpt"
    try:
        trainer.save_checkpoint(final_path, weights_only=True)
    except TypeError:
        # Lightning 2.2 lacks the explicit weights_only argument. The model's
        # state_dict is still saved, together with the normal trainer metadata.
        trainer.save_checkpoint(final_path)
    print(f"\nSFT complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
