# AutoVLA SFT on NVIDIA PAI AV

Single-GPU supervised fine-tuning of AutoVLA (Qwen2.5-VL-3B plus action
tokens) on the locally downloaded
`nvidia/PhysicalAI-Autonomous-Vehicles` dataset. The output checkpoint is a
warm start for later GRPO training in alpagym.

## What this implementation guarantees

- One shared processor is used by the dataset, collator and model. The script
  verifies that every `<action_N>` is one token and that the IDs begin at the
  configured `action_start_id`.
- Qwen's built-in causal-language-model loss is used, including the required
  one-token label shift.
- PAI timestamps are interpreted as microseconds. Four context frames end at a
  common anchor time, and the ten future waypoints are measured relative to the
  ego pose at that anchor.
- Trajectories are matched with the same final-contour algorithm used by
  AutoVLA's `TokenProcessor._match_agent_token`.
- The published `autovla.vlm.*`, legacy `drivevla.vlm.*`, and lightweight
  `vlm.*` checkpoint key formats are supported. Training stops if checkpoint
  coverage is below 95%.
- Train and validation clips are deterministic and disjoint. Only clips whose
  required camera and egomotion chunks exist locally are selected.
- Camera frames stay in memory; the loader does not create temporary JPEGs.

## Requirements

Use the existing alpagym virtual environment on the A100 server. Do not run the
script with the ambient conda environment's `python` or `pip`:

```text
/data/mnt_m62/10_personal/z59900495/workspace/alpagym/.venv/bin/python
```

That environment already contains the CUDA/PyTorch, Transformers,
FlashAttention, `physical_ai_av` and Qwen-VL dependencies used by the working
launch command. `physical_ai_av` itself requires Python 3.11 or newer. If a
dependency ever needs to be changed, use the same interpreter with
`.../.venv/bin/python -m pip`, not bare `pip`.

The codebook is the only runtime artifact this loader reads from the AutoVLA
source tree; it does not import navsim or nuplan. `AUTOVLA_REPO_PATH` is still
exported below to preserve the server's established launch environment.

## Configure paths

Edit these fields in `config/pai_sft.yaml` before running:

```yaml
model:
  pretrained_model_path: /path/to/Qwen2.5-VL-3B-Instruct
  sft_model_path: /path/to/AutoVLA_PDMS_89.ckpt
  codebook_cache_path: /path/to/AutoVLA/codebook_cache/agent_vocab.pkl

data:
  pai_data_dir: /data/mnt_m181/z59900495/workspace/DownloadTool-master/pai_dataset
```

The PAI root must contain `clip_index.parquet`, `features.csv`, metadata, and
the downloaded chunk files for these four features:

- `camera_front_wide_120fov`
- `camera_cross_left_120fov`
- `camera_cross_right_120fov`
- `egomotion`

## Run in three stages

From the repository root, first select the exact server interpreter and export
the existing AutoVLA path:

```bash
export AUTOVLA_REPO_PATH=/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA
PYTHON_BIN=/data/mnt_m62/10_personal/z59900495/workspace/alpagym/.venv/bin/python
```

Always start with the data-only preflight. It decodes a real batch and checks
the split, video inputs, trajectory and ten atomic action labels without
loading the 3B model:

```bash
"$PYTHON_BIN" run_sft.py --config config/pai_sft.yaml --preflight-only
```

Then run the bounded one-epoch training smoke test. This uses eight train clips
and four validation clips, regardless of the production sample limits in the
YAML:

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" run_sft.py \
  --config config/pai_sft.yaml \
  --smoke-test
```

Check the log for all of the following before starting the full job:

- `Action vocabulary ready` with IDs beginning at `151665`.
- `Data preflight passed` with ten action labels per batch item.
- A first waypoint that is not forcibly `(0, 0)` for moving clips.
- `Checkpoint loaded` with at least 95% parameter coverage.
- Finite `train_loss` and a reported `train_action_accuracy` in the smoke test.

Finally run the configured job:

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" run_sft.py --config config/pai_sft.yaml
```

The supplied production config uses every locally available training clip,
200 validation clips, batch size 1, gradient accumulation 4, BF16 mixed
precision, FlashAttention2 and gradient checkpointing. It hard-codes one
device by default. Change `training.devices` and `training.strategy` only after
the single-GPU run is correct; `strategy: auto` is not FSDP.

## Data timing and targets

With the default anchor of 2.0 seconds:

```text
camera context: 0.5s, 1.0s, 1.5s, 2.0s
current state:   2.0s
future targets:  2.5s, 3.0s, ..., 7.0s
```

Positions and headings are transformed into the ego frame at 2.0 seconds.
Velocity and acceleration are also taken at 2.0 seconds. The base PAI release
does not provide a route command, so the prompt explicitly says that navigation
is unavailable. It neither hard-codes every clip as `straight` nor derives a
command from the future trajectory, which would leak the training target into
the input.

The action matcher uses the last `[4, 2]` vehicle contour from every codebook
token, transforms it from the previous rollout state, and selects the token
whose four corners are closest to the ground-truth vehicle contour. This is
the deterministic `K=1` behavior of the original AutoVLA matcher.

## Outputs

Each run writes to `runs/sft/<timestamp>/`:

- the best validation checkpoints;
- `final.ckpt` containing model weights;
- the exact processor/tokenizer under `processor/`;
- CSV training metrics, including action-token accuracy.

The lightweight checkpoint stores `vlm.*` keys and can be loaded by the
AutoVLA/alpagym loader that removes optional `autovla.` or `drivevla.` wrapper
prefixes.

## Tests

Run tests with the same alpagym virtual environment:

```bash
"$PYTHON_BIN" -m unittest discover -s tests -v
```
