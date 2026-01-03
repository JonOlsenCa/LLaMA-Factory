# VGPT2 v4 Training Status Summary

## Current State (2026-01-02)
- **Repo:** C:\Github\LLM_fine-tuning
- **Hardware:** NVIDIA RTX A6000 (48GB VRAM), Windows 11, PowerShell 7+, Python 3.12, CUDA 12.8
- **Project:** Fine-tuning SQLCoder (defog/llama-3-sqlcoder-8b) on custom VGPT2 v4 dataset using LLaMA Factory

---

## Actions Taken
- Verified and activated Python venv: `.venv` in repo root
- Installed CUDA-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
- Resolved dependency pinning for numpy, pillow, fsspec
- Confirmed presence of key files:
    - `automation/configs/vgpt2_v4/stage1_sft.yaml` (training config)
    - `data/dataset_info.json` (dataset registry)
    - `data/vgpt2_v4_sft.json` or `data/vgpt2_v4_sft_expanded_clean.json` (actual data)
    - `training/01_start_sft_v4.ps1` (main SFT script)
- Fixed PowerShell script param block ordering and command invocation
- Validated CUDA, GPU, VRAM, and LLaMA Factory CLI availability

---

## What Worked
- Dry run and training script now launch without PowerShell errors
- CUDA, GPU, and VRAM detected correctly
- LLaMA Factory CLI runs and parses config
- Dataset registry (`dataset_info.json`) and config are present and correct
- Training script (`01_start_sft_v4.ps1`) is recognized and runs from any directory using absolute path

---

## What Hasn't Worked / Issues
- Initial errors: PowerShell param block placement, markdown link as path, missing CUDA wheel, dependency version conflicts
- Path resolution: Training script failed if run from wrong working directory or if `dataset_info.json` was missing/misplaced
- FileNotFoundError: If `data/dataset_info.json` is not found at runtime, training aborts

---

## Next Steps
1. **Ensure all data files exist:**
    - `data/dataset_info.json` (must be at repo root/data)
    - `data/vgpt2_v4_sft.json` or `data/vgpt2_v4_sft_expanded_clean.json` (must match config)
2. **Run training from repo root:**
    - Open PowerShell, activate venv, then:
      ```powershell
      pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File C:\Github\LLM_fine-tuning\training\01_start_sft_v4.ps1
      ```
3. **Monitor training:**
    - GPU: `nvidia-smi -l 1`
    - Logs: `Get-Content C:\Github\LLM_fine-tuning\saves\vgpt2_v4\sft\trainer_log.jsonl -Tail 10 -Wait`
4. **Validate model:**
    - After training, run:
      ```powershell
      C:\Github\LLM_fine-tuning\.venv\Scripts\python.exe C:\Github\LLM_fine-tuning\scripts\vgpt2_v4\probe.py --model C:\Github\LLM_fine-tuning\saves\vgpt2_v4\sft --output C:\Github\LLM_fine-tuning\output\probe_sft.json
      ```
5. **Chat test:**
    - ```powershell
      C:\Github\LLM_fine-tuning\.venv\Scripts\llamafactory-cli.exe chat --model_name_or_path C:\Github\LLM_fine-tuning\saves\vgpt2_v4\sft --template llama3 --adapter_name_or_path C:\Github\LLM_fine-tuning\saves\vgpt2_v4\sft
      ```
6. **If needed, run DPO stage:**
    - ```powershell
      pwsh -File C:\Github\LLM_fine-tuning\training\02_start_dpo_v4.ps1
      ```

---

## Script Locations
- **Main SFT script:** `training/01_start_sft_v4.ps1`
- **Config:** `automation/configs/vgpt2_v4/stage1_sft.yaml`
- **Dataset registry:** `data/dataset_info.json`
- **Data:** `data/vgpt2_v4_sft.json` or `data/vgpt2_v4_sft_expanded_clean.json`
- **Probe/validation:** `scripts/vgpt2_v4/probe.py`
- **DPO stage:** `training/02_start_dpo_v4.ps1`

---

## Definition of Success
- Training completes without error and produces a checkpoint in `saves/vgpt2_v4/sft/`
- Probe script runs and produces a valid output file
- Chat CLI works and generates sensible SQL responses
- (Optional) DPO stage runs and improves model preference alignment
- All steps reproducible from repo root with documented commands

---

## Quick Reference: How to Run
```powershell
# Activate venv
C:\Github\LLM_fine-tuning\.venv\Scripts\Activate.ps1

# Start SFT training
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File C:\Github\LLM_fine-tuning\training\01_start_sft_v4.ps1

# Monitor
nvidia-smi -l 1
Get-Content C:\Github\LLM_fine-tuning\saves\vgpt2_v4\sft\trainer_log.jsonl -Tail 10 -Wait

# Validate
C:\Github\LLM_fine-tuning\.venv\Scripts\python.exe C:\Github\LLM_fine-tuning\scripts\vgpt2_v4\probe.py --model C:\Github\LLM_fine-tuning\saves\vgpt2_v4\sft --output C:\Github\LLM_fine-tuning\output\probe_sft.json

# Chat test
C:\Github\LLM_fine-tuning\.venv\Scripts\llamafactory-cli.exe chat --model_name_or_path C:\Github\LLM_fine-tuning\saves\vgpt2_v4\sft --template llama3 --adapter_name_or_path C:\Github\LLM_fine-tuning\saves\vgpt2_v4\sft

# DPO stage (if needed)
pwsh -File C:\Github\LLM_fine-tuning\training\02_start_dpo_v4.ps1
```
