# VGPT2 v3 Training Scripts

Run these scripts in order from this directory.

## Quick Start

```powershell
cd c:\Github\LLM_fine-tuning

# 1. Review what's running
.\training\run\00_review_resources.ps1

# 2. Kill resource hogs (Chrome, Teams, etc.)
.\training\run\01_stop_resource_hogs.ps1

# 3. Start monitor in SEPARATE terminal
.\training\run\02_start_monitor.ps1

# 4. Start SFT training (main terminal)
.\training\run\03_train_sft.ps1

# 5. After SFT completes, run DPO
.\training\run\04_train_dpo.ps1

# 6. Optional: KTO training
.\training\run\05_train_kto.ps1

# 7. View all checkpoints
.\training\run\06_view_checkpoints.ps1
```

## Resource Optimization (RTX A6000 48GB)

All configs are optimized to maximize GPU utilization:

| Setting | Value | Rationale |
|---------|-------|-----------|
| `per_device_train_batch_size` | 2 | Uses ~45GB VRAM (94% utilization) |
| `gradient_accumulation_steps` | 8 | Effective batch size = 16 |
| `gradient_checkpointing` | true | Trades compute for memory |
| `dataloader_num_workers` | 0 | Required for Windows compatibility |
| `preprocessing_num_workers` | 16 | Utilizes Threadripper cores during data loading |

**Why CPU usage is low during training:**
- CPUs are only used during initial data loading/preprocessing
- Training is 100% GPU-bound (forward/backward passes happen on GPU)
- This is normal and expected for LLM training

## Scripts

| Script | Purpose |
|--------|---------|
| `00_review_resources.ps1` | Check GPU, RAM, CPU, disk space |
| `01_stop_resource_hogs.ps1` | Kill Chrome, Teams, Slack, etc. |
| `02_start_monitor.ps1` | Live monitor (run in separate terminal) |
| `03_train_sft.ps1` | Stage 1: Supervised Fine-Tuning |
| `04_train_dpo.ps1` | Stage 2: Direct Preference Optimization |
| `05_train_kto.ps1` | Stage 3: KTO (optional) |
| `06_view_checkpoints.ps1` | List all saved checkpoints |

## Checkpoint Locations

- **SFT**: `saves/vgpt2_v3/sft/`
- **DPO**: `saves/vgpt2_v3/dpo/`
- **KTO**: `saves/vgpt2_v3/final/`

## Training Data

- **SFT**: `data/vgpt2_v3_sft_merged.json` (70,006 examples)
- **DPO**: `data/vgpt2_v3_dpo.json` (9,500 pairs)
- **KTO**: `data/vgpt2_v3_kto.json`

## Configs

All configs in: `automation/configs/vgpt2_v3/`

## Resume Training (Automatic!)

**All configs now have `resume_from_checkpoint: true` by default.**

If training is interrupted:
1. Just re-run the same training script
2. It will automatically detect existing checkpoints
3. Training resumes from the latest checkpoint

**Maximum work lost on interruption:** ~15 minutes (200 steps between checkpoints)

## Checkpoint Safety

| Setting | Value | Protection |
|---------|-------|------------|
| `save_steps` | 200 | Checkpoint every ~15 min |
| `save_total_limit` | 3 | Keeps last 3 checkpoints |
| `overwrite_output_dir` | false | Never overwrites existing checkpoints |
| `resume_from_checkpoint` | true | Always resumes if interrupted |

## Expected Timeline

| Stage | Duration | Checkpoints | Max Work Lost |
|-------|----------|-------------|---------------|
| SFT | 8-12 hours | ~52 | ~15 min |
| DPO | 1-2 hours | ~6 | ~15 min |
| KTO | 30-60 min | ~3 | ~15 min |

