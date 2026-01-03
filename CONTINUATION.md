# VGPT2 V4 Continuation Notes (2026-01-01)

## Current State
- Training run completed successfully (no resume checkpoint in saves/vgpt2_v4/sft yet; fresh run finished).
- Environment: Python 3.12 venv (.venv312); torch/torchvision/torchaudio cu128; CUDA visible on RTX A6000.
- hf_xet optional extra documented; symlink warnings can be silenced with HF_HUB_DISABLE_SYMLINKS_WARNING=1.

## Key Files
- Training script: training/01_start_sft_v4.ps1
- Config: automation/configs/vgpt2_v4/stage1_sft.yaml
- Dataset: data/vgpt2_v4_sft_expanded_clean.json
- Docs: training/V4_TRAINING.md
- Env notes: requirements.txt (install-order with hf_xet note)

## Next Steps (when back)
1) Validate SFT output:
   - python scripts/probe_model.py --model saves/vgpt2_v4/sft
2) Quick chat test:
   - llamafactory-cli chat --model_name_or_path saves/vgpt2_v4/sft --template llama3 --adapter_name_or_path saves/vgpt2_v4/sft
3) If quality needs alignment: run DPO stage when ready:
   - .\training\02_start_dpo_v4.ps1

## If Training Needs Re-run
- Start fresh: .\training\01_start_sft_v4.ps1
- Resume from latest checkpoint: .\training\01_start_sft_v4.ps1 -Resume
- Dry run preview: .\training\01_start_sft_v4.ps1 -DryRun

## Environment Reminders
- Use Python 3.12 (avoid 3.14 due to datasets/dill Pickler error).
- Install HF accelerated cache (optional): pip install "huggingface_hub[hf_xet]"; set HF_HUB_DISABLE_SYMLINKS_WARNING=1 if symlink warnings persist on Windows.
- CUDA check: python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

## Open Items
- None blocking; proceed to validation and optional DPO.
