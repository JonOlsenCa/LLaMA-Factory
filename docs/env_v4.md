# V4 Environment Setup (Windows, CUDA)

## Prereqs
- Python 3.12.x
- CUDA drivers/toolkit matching torch cu128 wheels
- Git + PowerShell

## Steps
1) Create venv
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2) Install GPU torch (CUDA 12.8)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3) Install project deps (torch excluded)
```
pip install -r requirements.v4.txt
```

4) Sanity check CUDA
```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: cuda available = True and your GPU name.

## Training/Validation flow
- Dry-run SFT: `pwsh -NoProfile -File training/01_start_sft_v4.ps1 -DryRun`
- Train SFT:   `pwsh -NoProfile -File training/01_start_sft_v4.ps1`
- Probe:       `python scripts/vgpt2_v4/probe.py --model saves/vgpt2_v4/sft --output output/probe_sft.json`
- (Optional) DPO: `pwsh -NoProfile -File training/02_start_dpo_v4.ps1`

## Notes
- Keep torch install separate to avoid pulling CPU wheels.
- If you see CUDA not available in the script, reinstall torch with the cu128 wheel inside the venv.
- If symlink warnings appear on Windows with HF cache, optional: `pip install "huggingface_hub[hf_xet]"` and set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
