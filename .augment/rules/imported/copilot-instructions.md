# GitHub Copilot Instructions for LLaMA Factory Fine-Tuning

> **Hardware**: NVIDIA RTX A6000 (48GB) • 128GB RAM • AMD Threadripper 7960X (24c/48t) • Windows 11 + PowerShell 7+
>
> **Prime Directive**: Create PERFECT fine-tuned models. Cost, time, and complexity are NOT constraints. MAXIMIZE QUALITY.

---

## Enforcement Priority Key

| Icon | Level | Meaning |
|------|-------|---------|
| 🔴 | **Non-negotiable** | Violation = broken output. No exceptions. |
| 🟡 | **Strong defaults** | Override only with documented justification |
| 🟢 | **Best practices** | Follow unless conflicting with higher priority |

---

## Table of Contents

1. [Core Philosophy](#-core-philosophy)
2. [Pre-Action Verification](#-pre-action-verification)
3. [Windows/PowerShell Requirements](#-windowspowershell-requirements)
4. [A6000 Hardware Optimization](#-a6000-48gb-optimization)
5. [Dataset Pipeline](#-dataset-pipeline)
6. [Training Configuration](#-training-configuration)
7. [Failure Recovery](#-failure-recovery)
8. [Evaluation Protocol](#-evaluation-protocol)
9. [Experiment Tracking](#-experiment-tracking)
10. [Code Standards](#-code-standards)
11. [Project Structure](#-project-structure)
12. [Model-Specific Notes](#-model-specific-notes)
13. [Recommendation Format](#-recommendation-format)
14. [Quick Reference Checklists](#-quick-reference-checklists)

---

## 🔴 CORE PHILOSOPHY

### Zero Tolerance for Suboptimal Solutions

**This is production-grade work. Every shortcut compounds into technical debt.**

#### NEVER Suggest
- "Slightly slower" alternatives
- "Good enough" solutions
- "Fallback" options without exhausting the optimal path first
- "Optional" optimizations — if it's better, it's **REQUIRED**
- Workarounds that avoid solving the real problem

#### ALWAYS Do
- Fix the actual problem, not symptoms
- Install correct dependencies, never skip them
- Use the fastest/best implementation available
- Exhaust ALL options to make optimal solutions work before mentioning alternatives

#### When Something Fails
```
1. Diagnose the root cause
2. Research the correct fix
3. Implement the fix
4. Verify it works
5. ONLY if steps 1-4 fail after multiple serious attempts → ask user for guidance
```

#### Banned Phrases
```
❌ "slightly slower but works"
❌ "optional but recommended"
❌ "if that fails, you can..."
❌ "as a workaround..."
❌ "good enough for now"
❌ "we can skip this"
❌ "a simpler alternative..."
```

### Decision Framework
```
Quality Option (10x cost, 2x benefit)     → ✅ CHOOSE THIS
Efficient Option (1x cost, 1x benefit)    → ❌ REJECT

Higher LoRA rank vs. faster training      → Higher rank
Longer context vs. larger batch           → Longer context
More epochs vs. quicker iteration         → More epochs (for final models)
Debug the issue vs. use a workaround      → Debug the issue
```

---

## 🔴 PRE-ACTION VERIFICATION

**Before ANY file operation, command, or code generation, complete these checks IN ORDER:**

### 1. File/Path Verification
| Before... | Action Required |
|-----------|-----------------|
| Referencing any path | Use `view` tool to confirm existence |
| Creating/renaming files | Check if target already exists |
| Suggesting imports | Verify module exists in codebase |
| Modifying configs | Read current file content first |

### 2. Interface Verification
| Before... | Action Required |
|-----------|-----------------|
| Suggesting CLI flags | Inspect argparse/click definitions in script |
| Calling functions | Verify function signature in source |
| Using config keys | Check schema in `src/llamafactory/hparams/*.py` |
| Using dataset formats | Verify against `data/dataset_info.json` |

### 3. Version Verification
| Before... | Action Required |
|-----------|-----------------|
| Suggesting features | Check LLaMA Factory version in `pyproject.toml` |
| Using model features | Verify model supports them (e.g., flash attention) |
| Using training methods | Confirm method exists in current version |

### Violation Response
If verification is not possible:
> *"I cannot verify [X] exists. Please confirm before proceeding."*

---

## 🔴 WINDOWS/POWERSHELL REQUIREMENTS

### Command Execution Policy
- **NEVER** execute commands programmatically via `launch-process`
- **ALWAYS** provide commands for user to copy/paste
- **ALWAYS** assume PowerShell 7+ (not bash, not cmd.exe)

### Command Format Template
```powershell
# [Brief description]
# ⚠️ REQUIRES ADMIN (only if true)
# ⏱️ Estimated time: X minutes (for long operations)
cd C:\Github\LLM_fine-tuning
.\path\to\script.ps1 -Flag value
```

### PowerShell Regex Syntax (.NET)
```powershell
# ❌ WRONG - bash-style escaped OR
Select-String -Pattern "argparse\|add_argument"

# ✅ CORRECT - .NET regex OR
Select-String -Pattern "argparse|add_argument"

# ❌ WRONG - grep doesn't exist
Get-Content file.txt | grep "pattern"

# ✅ CORRECT - PowerShell equivalent
Get-Content file.txt | Select-String -Pattern "pattern"

# ✅ CORRECT - Recursive search
Get-ChildItem -Recurse -Filter "*.py" | Select-String -Pattern "pattern"
```

### 🔴 NO ANGLE BRACKET PLACEHOLDERS
PowerShell interprets `<` as a reserved operator.

```powershell
# ❌ WRONG - Causes PowerShell parse error
python script.py --config <your_config.yaml>

# ✅ CORRECT - Use actual paths
python script.py --config automation/configs/vgpt2_v3/stage2_dpo.yaml

# ✅ CORRECT - Use ALL_CAPS with example
# Replace CONFIG_PATH with your config file
python script.py --config CONFIG_PATH
# Example: python script.py --config automation/configs/vgpt2_v3/stage2_dpo.yaml
```

### 🔴 Python Snippet Execution
```powershell
# ❌ WRONG - Bash heredoc doesn't exist
python <<'PY'
import json
print(json.dumps({"key": "value"}))
PY

# ❌ WRONG - Quote issues
python -c "import json; print(json.dumps({"key": "value"}))"

# ✅ CORRECT - Single quotes inside double-quoted -c
python -c "import json; print(json.dumps({'key': 'value'}))"

# ✅ CORRECT - Here-string for multi-line
@"
import json
data = {"key": "value"}
print(json.dumps(data, indent=2))
"@ | python
```

### 🔴 File Encoding
**Every file operation MUST specify UTF-8:**

```python
# ✅ REQUIRED - All file operations
with open("file.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(content)

Path("file.txt").read_text(encoding="utf-8")
Path("file.txt").write_text(content, encoding="utf-8")

pd.read_csv("data.csv", encoding="utf-8")

# ❌ WILL FAIL - Missing encoding
with open("file.json", "r") as f:  # UnicodeDecodeError
    data = json.load(f)
```

---

## 🔴 A6000 48GB OPTIMIZATION

### Always-On Settings
**Include in EVERY training config without exception:**

```yaml
bf16: true                      # A6000 has native bf16 - NEVER use fp16
flash_attn: fa2                 # 2x memory efficiency, faster training
gradient_checkpointing: true    # Unless batch size already very small
torch_compile: false            # Often slower for fine-tuning
```

### Memory Budget Reference

| Context Length | LoRA r=256 | LoRA r=512 | Full Finetune |
|----------------|------------|------------|---------------|
| 4096           | ~24 GB     | ~32 GB     | OOM           |
| 8192           | ~36 GB     | ~44 GB     | OOM           |
| 16384          | ~46 GB     | OOM        | OOM           |
| 32768          | OOM        | OOM        | OOM           |

*Values assume 7B parameter model with gradient checkpointing enabled*

### Batch Size Optimization
```yaml
# Goal: Fill VRAM to ~95% utilization
# Use gradient_accumulation_steps to achieve effective batch size

# Example for 8192 context, LoRA r=256:
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
# Effective batch size: 2 × 8 = 16
```

### DeepSpeed Configuration
```yaml
# Stage 2: Use when batch size constrained at long context
deepspeed: examples/deepspeed/ds_z2_config.json

# Stage 3: Rarely needed for single A6000, adds overhead
# Only use if Stage 2 still OOMs
```

### CPU Offload (Last Resort Only)
```yaml
# Significant speed penalty - use only when necessary
offload_model: true
```

---

## 🔴 DATASET PIPELINE

### Pre-Training Validation (MANDATORY)
**Before ANY training run, validate dataset loading:**

```powershell
cd C:\Github\LLM_fine-tuning
@"
from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset

model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(
    dict(config='automation/configs/YOUR_CONFIG.yaml')
)
dataset = get_dataset(model_args, data_args, training_args, stage=finetuning_args.stage)
print(f'✓ Loaded {len(dataset)} samples')
print(f'✓ Columns: {dataset.column_names}')
"@ | python
```

### Supported Dataset Formats

| Format | Required Fields | Stage |
|--------|-----------------|-------|
| ShareGPT | `conversations: [{from, value}]` | SFT |
| Alpaca | `instruction, input, output` | SFT |
| DPO/ORPO | `conversations, chosen, rejected` | DPO/ORPO |
| KTO | `conversations, label` (bool) | KTO |
| Pretrain | `text` | PT |

### Dataset Registration
All datasets must be registered in `data/dataset_info.json`:

```json
{
  "my_dataset": {
    "file_name": "path/to/data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
```

### Data Quality Checklist
- [ ] Dataset loads without errors (run validation above)
- [ ] No truncation warnings at target `cutoff_len`
- [ ] Token distribution is balanced
- [ ] No duplicate samples
- [ ] Response quality spot-checked manually (10-20 examples)
- [ ] Special tokens handled correctly for model family

### ShareGPT Format Variations
```json
// ✅ CORRECT - Standard format
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

// ❌ WRONG - "role" instead of "from"
{"conversations": [{"role": "user", "content": "..."}]}

// ❌ WRONG - "messages" instead of "conversations"
{"messages": [{"from": "human", "value": "..."}]}
```

### Problematic Samples to Filter
```python
# These cause training issues - filter them out
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": ""}]}     # Empty
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "OK"}]}   # Too short
```

---

## 🔴 TRAINING CONFIGURATION

### Mandatory Config Template
**Every training config MUST include all sections:**

```yaml
### Model Configuration
model_name_or_path: Qwen/Qwen2.5-7B-Instruct  # Full HF path or local path
trust_remote_code: true
flash_attn: fa2
bf16: true

### Dataset Configuration
dataset: your_dataset_name       # Must exist in dataset_info.json
template: qwen                   # Must match model family
cutoff_len: 8192                 # Start high, reduce only if OOM
preprocessing_num_workers: 16    # Utilize Threadripper

### Training Hyperparameters
stage: sft                       # sft, dpo, orpo, kto, pt, rm, ppo
do_train: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
gradient_checkpointing: true
max_grad_norm: 1.0

### LoRA Configuration
finetuning_type: lora
lora_rank: 256                   # Start at 256, NOT 8 or 16
lora_alpha: 512                  # Typically 2x lora_rank
lora_dropout: 0.05
lora_target: all                 # Target all linear layers

### Checkpointing (NEVER SKIP)
output_dir: saves/experiment_name_YYYYMMDD
save_steps: 250
save_total_limit: 5
save_only_model: false           # Keep optimizer state for resume
logging_steps: 10

### Evaluation (if eval dataset provided)
eval_strategy: steps
eval_steps: 250
per_device_eval_batch_size: 4

### Reporting
report_to: wandb
run_name: descriptive_name
```

### Quality-First Defaults

| Parameter | Minimum Value | Rationale |
|-----------|---------------|-----------|
| `lora_rank` | 256 | Higher capacity for complex patterns |
| `lora_alpha` | 512 | Standard 2x ratio |
| `cutoff_len` | 8192 | Utilize full context |
| `num_train_epochs` | 3 | Thorough learning |
| `learning_rate` | 2e-4 | Standard for LoRA |
| `warmup_ratio` | 0.1 | Stable training start |

---

## 🔴 FAILURE RECOVERY

### OOM Recovery Hierarchy
**Try in this exact order. Do NOT skip steps.**

| Step | Action | Quality Impact |
|------|--------|----------------|
| 1 | Reduce `per_device_train_batch_size` by 50% | None |
| 2 | Enable `gradient_checkpointing: true` | None |
| 3 | Reduce `cutoff_len` to 4096 | ⚠️ Document tradeoff |
| 4 | Lower `lora_rank` to 128 | ⚠️ Document tradeoff |
| 5 | Enable DeepSpeed Stage 2 | Minor overhead |
| 6 | Enable `offload_model: true` | ⚠️ Significant slowdown |

```yaml
# OOM-safe configuration
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
cutoff_len: 4096
lora_rank: 128
deepspeed: examples/deepspeed/ds_z2_config.json
```

### NaN/Loss Explosion Recovery

| Step | Action |
|------|--------|
| 1 | Reduce `learning_rate` by 10x (2e-4 → 2e-5) |
| 2 | Adjust `max_grad_norm: 0.5` |
| 3 | Increase `warmup_ratio: 0.2` |
| 4 | Check dataset for malformed samples |
| 5 | Verify tokenizer matches base model exactly |

```powershell
# Find problematic samples
cd C:\Github\LLM_fine-tuning
@"
import json
with open('data/your_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for i, item in enumerate(data):
    if 'conversations' in item:
        for turn in item['conversations']:
            if len(turn.get('value', '')) < 5:
                print(f'Sample {i}: Short/empty turn: {turn}')
"@ | python
```

### Resume from Checkpoint
```yaml
# Resume after crash/interruption
resume_from_checkpoint: C:/Github/LLM_fine-tuning/saves/experiment_name/checkpoint-1000
ignore_data_skip: false

# If optimizer states corrupted, start fresh from model checkpoint
resume_from_checkpoint: false
model_name_or_path: C:/Github/LLM_fine-tuning/saves/experiment_name/checkpoint-1000
```

### Common Error Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | VRAM exceeded | Follow OOM hierarchy |
| `expected scalar type BFloat16` | Mixed precision mismatch | Ensure `bf16: true` everywhere |
| `KeyError: 'conversations'` | Dataset format mismatch | Check `formatting` in dataset_info.json |
| `Token indices sequence length` | Exceeds cutoff | Increase `cutoff_len` or truncate data |
| `loss: nan` | Training instability | Follow NaN recovery |
| `No module named 'flash_attn'` | Missing dependency | `pip install flash-attn --no-build-isolation` |

---

## 🟡 EVALUATION PROTOCOL

### Mandatory Evaluation Points
- ✅ After every training stage completion
- ✅ Before deploying any checkpoint
- ✅ When loss curve plateaus unexpectedly
- ✅ Before and after merging LoRA adapters

### Quick Sanity Check
```powershell
cd C:\Github\LLM_fine-tuning
python -m llamafactory.chat.cli `
    --model_name_or_path saves/YOUR_CHECKPOINT `
    --adapter_name_or_path saves/YOUR_CHECKPOINT `
    --template qwen `
    --finetuning_type lora
```

### Batch Evaluation
```powershell
cd C:\Github\LLM_fine-tuning
llamafactory-cli eval `
    --model_name_or_path saves/YOUR_CHECKPOINT `
    --adapter_name_or_path saves/YOUR_CHECKPOINT `
    --template qwen `
    --finetuning_type lora `
    --task mmlu `
    --split test `
    --batch_size 4
```

### Evaluation Metrics by Stage

| Stage | Primary Metrics | Watch For |
|-------|-----------------|-----------|
| SFT | Loss, perplexity | Overfitting (eval >> train loss) |
| DPO | Accuracy, reward margin | Reward hacking |
| ORPO | Loss, odds ratio | Collapse to single response style |
| KTO | Loss, accuracy | Class imbalance issues |

---

## 🟡 EXPERIMENT TRACKING

### Naming Convention
```
{model}_{method}_{dataset}_{key_param}_{date}

Examples:
qwen2.5-7b_lora256_vgpt2v3_lr2e4_20250102
llama3-8b_dpo_preferences_b16_20250102
mistral-7b_orpo_combined_r512_20250102
```

### Required Tracking Config
```yaml
output_dir: saves/qwen2.5-7b_lora256_vgpt2v3_lr2e4_20250102
report_to: wandb
run_name: qwen2.5-7b_lora256_vgpt2v3_lr2e4
logging_steps: 10
logging_first_step: true
save_steps: 250
eval_steps: 250
```

### Experiment Log Template
Maintain `experiments/log.md`:

```markdown
## 2025-01-02: Qwen2.5-7B LoRA SFT

**Config**: `automation/configs/vgpt2_v3/stage1_sft.yaml`
**Dataset**: vgpt2_v3_sft (15,000 samples)
**Duration**: 4.5 hours

### Hyperparameters
- LoRA rank: 256, alpha: 512
- Learning rate: 2e-4, cosine schedule
- Batch size: 2 × 8 = 16 effective
- Context: 8192 tokens

### Results
- Final loss: 0.85
- Eval loss: 0.92
- VRAM usage: 44 GB peak

### Observations
- Loss plateau after epoch 2
- Consider higher rank for next run

### Next Steps
- [ ] Run DPO stage with this checkpoint
- [ ] Compare against r=512 baseline
```

---

## 🟢 CODE STANDARDS

### Style Requirements
- **Guide**: Google Python Style Guide
- **Linter**: Ruff
- **Line length**: 119 characters
- **Quotes**: Double quotes
- **Docstrings**: Google style

### Import Order
```python
# Standard library
import json
import os
import sys
from pathlib import Path

# Third-party
import torch
import transformers
from accelerate import Accelerator
from peft import LoraConfig

# First-party
from llamafactory.data import get_dataset
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model
```

### Type Hints (Required)
```python
def process_dataset(
    data_path: str | Path,
    max_samples: int | None = None,
    shuffle: bool = True,
) -> list[dict[str, Any]]:
    """Process dataset from file.

    Args:
        data_path: Path to the dataset file.
        max_samples: Maximum samples to load. None for all.
        shuffle: Whether to shuffle the data.

    Returns:
        List of processed samples as dictionaries.

    Raises:
        FileNotFoundError: If data_path doesn't exist.
        ValueError: If dataset format is invalid.
    """
    ...
```

### Quality Commands
```powershell
cd C:\Github\LLM_fine-tuning
make style      # Auto-fix formatting
make quality    # Run all linters
make test       # Run test suite
make commit     # Run all pre-commit hooks
make license    # Check license headers
```

---

## 🟢 PROJECT STRUCTURE

### LLaMA Factory Core
```
src/llamafactory/
├── api/          # OpenAI-compatible API server
├── chat/         # Interactive chat CLI
├── data/         # Dataset loading and processing
├── eval/         # Evaluation utilities
├── extras/       # Logging, callbacks, constants
├── hparams/      # Hyperparameter dataclasses (CHECK HERE FOR CONFIG KEYS)
├── model/        # Model loading, patching, quantization
├── train/        # Training pipelines
│   ├── sft/      # Supervised fine-tuning
│   ├── dpo/      # Direct preference optimization
│   ├── orpo/     # Odds ratio preference optimization
│   ├── kto/      # Kahneman-Tversky optimization
│   ├── ppo/      # Proximal policy optimization
│   ├── rm/       # Reward modeling
│   └── pt/       # Pretraining
└── webui/        # Gradio interface
```

### Automation Structure
```
automation/
├── configs/           # Training configurations by project
│   └── vgpt2_v3/
│       ├── stage1_sft.yaml
│       ├── stage2_dpo.yaml
│       └── stage3_orpo.yaml
├── scripts/           # Utility scripts
├── eval/              # Evaluation prompts and results
└── experiments/       # Experiment logs
    └── log.md
```

### Key Entry Points
```powershell
# Training
llamafactory-cli train --config path/to/config.yaml

# Interactive chat
llamafactory-cli chat --model_name_or_path MODEL --template TEMPLATE

# Web UI
llamafactory-cli webui

# API server
llamafactory-cli api --model_name_or_path MODEL --template TEMPLATE

# Evaluation
llamafactory-cli eval --model_name_or_path MODEL --task TASK

# Export/merge LoRA
llamafactory-cli export --model_name_or_path MODEL --adapter_name_or_path ADAPTER --export_dir OUTPUT
```

---

## 🟢 MODEL-SPECIFIC NOTES

### Qwen2.5
```yaml
template: qwen          # NOT chatml
flash_attn: fa2         # Works, requires: pip install flash-attn --no-build-isolation
```

### LLaMA 3 / 3.1
```yaml
template: llama3        # NOT llama2
add_bos_token: false    # LLaMA 3 adds it automatically
```

### Mistral
```yaml
template: mistral
# Sliding window attention may need adjustment for very long contexts
```

### LoRA Merging
```powershell
# ALWAYS test before and after merging
llamafactory-cli export `
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct `
    --adapter_name_or_path saves/lora_checkpoint `
    --template qwen `
    --finetuning_type lora `
    --export_dir saves/merged_model

# Test immediately
python -m llamafactory.chat.cli --model_name_or_path saves/merged_model --template qwen
```

### DPO After SFT
```yaml
# MUST use SFT checkpoint as base
model_name_or_path: saves/sft_checkpoint  # NOT original model
adapter_name_or_path: null                 # Fresh LoRA for DPO
stage: dpo
pref_beta: 0.1
pref_loss: sigmoid
```

---

## 🔴 RECOMMENDATION FORMAT

**When presenting options, use this structure:**

```markdown
## Objective
[State what we're optimizing for]

## Options (Ranked by Expected Quality)

### 🥇 Option 1: [Best Quality]
- **Quality Impact**: [Expected outcome]
- **Resource Usage**: GPU: X%, VRAM: Y GB, Time: Z hrs
- **Tradeoffs**: [What this costs]
- **Config**: [Key settings]

### 🥈 Option 2: [Second Best]
...

### 🥉 Option 3: [Fallback]
...

## Recommendation
I recommend **Option [N]** because [quality-focused rationale].

[Provide complete config/commands for recommended option]
```

### Recommendation Rules
- **ALWAYS** rank by quality, best first
- **ALWAYS** make a clear recommendation
- **NEVER** present conservative options as defaults
- **NEVER** omit complex options because they're harder
- **NEVER** be neutral when one option is clearly better

---

## 🔴 QUICK REFERENCE CHECKLISTS

### Before Suggesting ANY Command
- [ ] Verified file/path exists with `view` tool
- [ ] Checked script interface (argparse flags)
- [ ] Used PowerShell syntax (not bash)
- [ ] No angle bracket placeholders
- [ ] Provided absolute paths from `C:\Github\LLM_fine-tuning`
- [ ] Stated if admin privileges or long runtime expected

### Before Recommending Training Config
- [ ] Included all mandatory sections
- [ ] Checkpointing configured (`save_steps`, `save_total_limit`)
- [ ] Maximized for quality (not efficiency)
- [ ] `lora_rank` ≥ 256
- [ ] `cutoff_len` ≥ 8192 (or documented why lower)
- [ ] `bf16` and `flash_attn` enabled
- [ ] `gradient_checkpointing` enabled

### Before Any File Operation
- [ ] Included `encoding='utf-8'` in all `open()` calls
- [ ] Used `Path` objects with `.read_text(encoding='utf-8')`
- [ ] Handled potential `FileNotFoundError`

### Before Presenting Options
- [ ] Ranked by quality (best first)
- [ ] Made clear recommendation with rationale
- [ ] Showed resource utilization for each
- [ ] Included ready-to-use config/commands for recommended option

---

*Last updated: 2025-01-03*
