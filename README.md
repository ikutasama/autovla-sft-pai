# AutoVLA SFT on PAI Dataset

SFT (supervised fine-tuning) of AutoVLA (Qwen2.5-VL-3B + action tokens) on the
NVIDIA PhysicalAI-Autonomous-Vehicles dataset. This produces a warm-start
checkpoint for GRPO RL training in alpagym.

## Why SFT first?

Direct GRPO from the PDMS SFT checkpoint produces poor trajectories (high
variance, some barely moving). SFT on PAI data teaches the model to generate
reasonable driving trajectories on real-world data, giving GRPO a better
starting point.

## Prerequisites

1. **AutoVLA repo** at `/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA`
   - Must have `codebook_cache/agent_vocab.pkl`
   - Must have `models/autovla.py` (SFTAutoVLA class)

2. **PAI dataset** downloaded locally:
   ```
   /data/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/pai_dataset/
   ```

3. **physical_ai_av** package installed (v0.2.2+):
   ```
   pip install physical_ai_av
   ```

4. **Qwen2.5-VL-3B-Instruct** model at:
   ```
   /data/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Qwen/Qwen2.5-VL-3B-Instruct
   ```

## Installation

```bash
cd /data/mnt_m62/10_personal/z59900495/workspace
git clone https://github.com/ikutasama/autovla-sft-pai.git
cd autovla-sft-pai
```

## Running SFT

```bash
export AUTOVLA_REPO_PATH=/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA

python run_sft.py --config config/pai_sft.yaml
```

### Key config options (in `config/pai_sft.yaml`)

| Field | Description |
|-------|-------------|
| `model.sft_model_path` | Starting checkpoint (AutoVLA_PDMS_89.ckpt) |
| `model.codebook_cache_path` | Action token codebook (agent_vocab.pkl) |
| `data.pai_data_dir` | PAI dataset root directory |
| `training.epochs` | Number of SFT epochs |
| `training.batch_size` | Batch size (1 for 80GB A100) |
| `training.train_sample_size` | Limit samples (null = all) |

## Data Pipeline

```
PAI clip
  ├── camera_front_wide_120fov  ─┐
  ├── camera_cross_left_120fov   ├──→ 4 frames @ 2Hz each → prompt
  ├── camera_cross_right_120fov ─┘
  └── egomotion labels
       ├── position (x, y, heading)
       ├── velocity, acceleration
       └── GT trajectory (10 waypoints @ 0.5s, 5s horizon)
            └── PAICodebookMatcher → 10 action token indices
                 └── ActionTokenizer → "<action_XXX>..." text
                      └── SFT target (cross-entropy loss)
```

## Using SFT checkpoint for GRPO

After SFT, copy the checkpoint to the alpagym model path:

```bash
cp runs/sft/<date>/final.ckpt /data/mnt_m62/.../AutoVLA/AutoVLA_SFT_PAI.ckpt
```

Then update the alpagym launch script to use the new checkpoint and run GRPO as before.

## Architecture

- `pai_sft_dataset.py`: PAI data loader + trajectory→action token converter
- `run_sft.py`: PyTorch Lightning + FSDP training script
- `config/pai_sft.yaml`: Configuration

## Notes

- The trajectory→action token conversion uses nearest-neighbor matching against
  the codebook (agent_vocab.pkl). This matches the AutoVLA navsim
  `_match_agent_token` logic but is simplified (contour-based matching is
  replaced with endpoint matching).
- Camera frames are extracted from PAI video zip files at 2Hz.
- The SFT prompt format is identical to AutoVLA's original SFTDataset, ensuring
  compatibility with the inference model.
