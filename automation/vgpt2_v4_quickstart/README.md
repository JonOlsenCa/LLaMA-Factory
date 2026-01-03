# VGPT2 v4 Training Quickstart

> **One-command training for ViewpointGPT v4 using SQLCoder-optimized approach**

## 📋 Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Why SQLCoder?](#why-sqlcoder)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Scripts Reference](#scripts-reference)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)

---

## Overview

**VGPT2 v4** is our SQL query generation model fine-tuned on Viewpoint Vista ERP schemas. This version uses:

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base Model | `defog/llama-3-sqlcoder-8b` | Pre-trained on 200K+ SQL examples |
| Method | LoRA (rank=128) | Memory-efficient, quality preserved |
| Dataset | 3,357 curated examples | Schema-in-prompt format |
| Framework | LlamaFactory | Production-ready training |

---

## Environment Setup

**Verified working configuration (January 2025):**

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12.10 | **Required** - flash-attn wheels only available for 3.12 |
| PyTorch | 2.5.1+cu124 | **Required** - must match flash-attn wheel |
| CUDA Toolkit | 12.6 | Installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6` |
| flash-attn | 2.7.4.post1 | Pre-built wheel from kingbri1/flash-attention |
| VS Build Tools | 2022 (v143) | For any native compilation needs |

### First-Time Setup Steps

**1. Install Prerequisites (one-time)**
```powershell
# Install Python 3.12
winget install Python.Python.3.12

# Install CUDA Toolkit 12.6 (if not installed)
# Download from: https://developer.nvidia.com/cuda-12-6-0-download-archive

# Install VS 2022 Build Tools with "Desktop development with C++"
# Download from: https://aka.ms/vs/17/release/vs_BuildTools.exe
```

**2. Create Environment**
```powershell
cd C:\Github\LLM_fine-tuning

# Create venv with Python 3.12
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch 2.5.1+cu124 (MUST match flash-attn wheel)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install LLaMA Factory
pip install -e ".[torch,metrics]"

# Install flash-attn from pre-built wheel (building from source fails on Windows/MSVC)
pip install wheel
pip install https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.5.1cxx11abiFALSE-cp312-cp312-win_amd64.whl

# Install extras
pip install wandb tensorboard
```

**3. Verify Installation**
```powershell
python -c "import torch; from flash_attn import flash_attn_func; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, flash-attn OK')"
```

Expected output:
```
PyTorch 2.5.1+cu124, CUDA 12.4, flash-attn OK
```

### Why These Specific Versions?

| Constraint | Reason |
|------------|--------|
| Python 3.12 | Only version with available flash-attn Windows wheels |
| PyTorch 2.5.1+cu124 | Flash-attn wheel compiled against this exact version |
| Pre-built wheel | Building flash-attn from source fails due to MSVC/cutlass template bugs |

---

## Why SQLCoder?

### The Problem with Generic LLMs
Generic models (LLaMA, Mistral, Qwen) learn SQL from scratch during fine-tuning. They:
- Hallucinate table/column names
- Generate syntactically invalid SQL
- Struggle with complex JOINs
- Need 50K+ examples to learn basics

### SQLCoder Advantage
`defog/llama-3-sqlcoder-8b` was specifically trained on SQL generation:

```
┌─────────────────────────────────────────────────────────────┐
│  Generic LLM                      SQLCoder                  │
│  ───────────                      ────────                  │
│  Learns SQL syntax    →    Already knows SQL syntax         │
│  Needs schema hints   →    Native schema understanding      │
│  3,000 examples       →    3,000 examples = domain expert   │
│  Accuracy: ~60%       →    Accuracy: ~85%+ expected         │
└─────────────────────────────────────────────────────────────┘
```

### Schema-in-Prompt Format
Our v4 dataset uses SQLCoder's native format:
```
Question: [Natural language question]

Database Schema:
CREATE TABLE APTH (
  APCo bcompany NOT NULL,
  ...
);

Provide:
1. A brief explanation
2. The SQL query
3. Vista-specific notes
```

This format enables the model to:
1. **Reference exact columns** from the provided DDL
2. **Apply Vista conventions** (WITH NOLOCK, company filters)
3. **Explain its reasoning** before generating SQL

---

## Hardware Requirements

This setup is optimized for:

| Component | Spec | Notes |
|-----------|------|-------|
| GPU | NVIDIA RTX A6000 (48GB) | Also works on 24GB with adjustments |
| RAM | 128GB | 64GB minimum |
| CPU | AMD Threadripper 7960X | Any modern CPU works |
| OS | Windows 11 | PowerShell 7+ |
| Storage | SSD with 50GB free | For model cache and checkpoints |

---

## Quick Start

### From Fresh Terminal (C:\Users\olsen) - ONE COMMAND

Just copy-paste this into PowerShell:

```powershell
& "C:\Github\LLM_fine-tuning\automation\vgpt2_v4_quickstart\01_train.ps1"
```

**That's it!** The script handles everything:
- ✅ Navigates to project directory
- ✅ Uses venv Python directly (no activation needed)
- ✅ Verifies GPU availability
- ✅ Runs training

Training will complete in ~30-45 minutes.

---

## Scripts Reference

| Script | Purpose | Duration |
|--------|---------|----------|
| `01_train.ps1` | Full SFT training | ~30-45 min |
| `02_chat.ps1` | Interactive testing | Manual |
| `03_evaluate.ps1` | Batch evaluation | ~5 min |

---

## Expected Results

### Training Metrics
- **Initial Loss**: ~2.0-2.5
- **Final Loss**: ~0.3-0.5
- **Steps**: ~200-400 (with packing)
- **VRAM Usage**: ~35-40GB peak

### Quality Indicators
After training, the model should:
- ✅ Use correct Vista table names (APTH, JCJM, etc.)
- ✅ Include WITH (NOLOCK) hints
- ✅ Apply company/job filters appropriately
- ✅ Generate valid SQL syntax
- ✅ Explain reasoning before SQL

---

## Troubleshooting

### "Flash Attention not found"

**Do NOT try `pip install flash-attn --no-build-isolation`** - it fails on Windows due to MSVC/cutlass template bugs.

Use the pre-built wheel:
```powershell
pip install https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.5.1cxx11abiFALSE-cp312-cp312-win_amd64.whl
```

**Requirements for pre-built wheel:**
- Python 3.12 (check: `python --version`)
- PyTorch 2.5.1+cu124 (check: `python -c "import torch; print(torch.__version__)"`)

If you have different versions, either:
1. Recreate venv with correct versions (recommended)
2. Use SDPA fallback: edit `stage1_sft.yaml` and change `flash_attn: fa2` to `flash_attn: sdpa`

### Out of Memory (OOM)
Reduce batch size in config:
```yaml
per_device_train_batch_size: 4  # from 8
gradient_accumulation_steps: 4  # from 2
```

### Slow Training (>10s/iteration)
1. Ensure `packing: true` is set
2. Check no other GPU processes: `nvidia-smi`
3. Verify VRAM isn't swapping

### PyTorch/CUDA Version Mismatch
If you see errors about CUDA version mismatches:
```powershell
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install correct version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

---

## File Locations

```
C:\Github\LLM_fine-tuning\
├── automation/
│   ├── configs/vgpt2_v4/
│   │   └── stage1_sft.yaml      ← Training config
│   └── vgpt2_v4_quickstart/
│       ├── README.md             ← This file
│       ├── 01_train.ps1          ← Training script
│       ├── 02_chat.ps1           ← Interactive chat
│       └── 03_evaluate.ps1       ← Batch evaluation
├── data/
│   └── vgpt2_v4_sft_expanded_clean.json  ← Training data
├── saves/
│   └── vgpt2_v4/sft_optimized/   ← Output checkpoints
└── venv/                         ← Python environment
```

