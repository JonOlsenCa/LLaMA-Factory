# VGPT2 v3 Training Scripts

All scripts needed to train and monitor VGPT2 v3.

## Quick Start

```powershell
cd C:\Github\LLM_fine-tuning

# Terminal 1: Start training
.\training\01_start_sft.ps1

# Terminal 2: Monitor GPU
.\training\monitor_gpu.ps1

# Terminal 3: Monitor checkpoints
.\training\monitor_checkpoints.ps1
```

## Training Stages

| Script | Purpose | Time |
|--------|---------|------|
| `01_start_sft.ps1` | Start Stage 1 SFT | 8-12 hours |
| `02_resume_sft.ps1` | Resume SFT from checkpoint | Remaining |
| `03_start_dpo.ps1` | Start Stage 2 DPO | 30-60 min |
| `04_start_kto.ps1` | Start Stage 3 KTO | 15-30 min |
| `05_validate.ps1` | Run validation tests | 5-15 min |

## Monitoring Scripts

| Script | Purpose |
|--------|---------|
| `monitor_gpu.ps1` | Real-time GPU stats (utilization, memory, temp) |
| `monitor_checkpoints.ps1` | Watch for new checkpoints, show progress |

## Checkpoint Schedule

**SFT (Stage 1):**
- Saves every 500 steps
- First checkpoint: ~20-30 minutes
- Keeps last 5 checkpoints
- Location: `saves/vgpt2_v3/sft/checkpoint-*`

**DPO (Stage 2):**
- Saves every 200 steps
- Location: `saves/vgpt2_v3/dpo/checkpoint-*`

**KTO (Stage 3):**
- Saves every 200 steps
- Final model: `saves/vgpt2_v3/final/`

## Testing Resume Functionality

1. Start training: `.\training\01_start_sft.ps1`
2. Wait for first checkpoint (~20-30 min)
3. Stop training: `Ctrl+C`
4. Verify checkpoint exists: `ls saves/vgpt2_v3/sft/`
5. Resume: `.\training\02_resume_sft.ps1`
6. Confirm it continues from checkpoint (check step number in output)

## Validation

```powershell
# Quick validation (fewer tests)
.\training\05_validate.ps1 -Stage sft -Quick

# Full validation
.\training\05_validate.ps1 -Stage final
```

## Troubleshooting

**Out of memory:**
- Reduce `per_device_train_batch_size` in config
- Ensure no other GPU apps running

**Training too slow:**
- Check GPU utilization with `monitor_gpu.ps1`
- Should be >90% utilization

**Resume not working:**
- Check checkpoint exists in `saves/vgpt2_v3/sft/`
- Ensure `checkpoint-*` folders have `pytorch_model.bin` or `adapter_model.safetensors`

