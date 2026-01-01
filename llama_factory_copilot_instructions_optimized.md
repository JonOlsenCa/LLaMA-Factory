# GitHub Copilot Instructions for LLaMA Factory

> **Enforcement Priority**: Rules marked 🔴 are non-negotiable. Rules marked 🟡 are strong defaults. Rules marked 🟢 are best practices.

---

## 🔴 MANDATORY PRE-ACTION CHECKS

Before ANY file operation, command suggestion, or code generation, complete these verification steps IN ORDER:

### 1. File/Path Verification
```
BEFORE referencing any path → Use `view` tool to confirm existence
BEFORE creating/renaming files → Check if target already exists
BEFORE suggesting imports → Verify module exists in codebase
```

### 2. Interface Verification
```
BEFORE suggesting CLI flags → Inspect argparse/click definitions in the script
BEFORE calling functions → Verify function signature and parameters
BEFORE using config keys → Check actual config schema in hparams/
```

### 3. Environment Context
```
ALWAYS assume Windows + PowerShell (not bash)
ALWAYS assume UTF-8 encoding is NOT default
ALWAYS provide absolute paths from workspace root
```

**Violation Response**: If you cannot verify, STATE that verification is needed and ask the user to confirm, rather than assuming.

---

## 🔴 WINDOWS/POWERSHELL REQUIREMENTS

### Never Execute Directly
Provide commands for user to copy/paste. Never use `launch-process` or execute PowerShell commands programmatically.

### Command Format Template
```powershell
# [Brief description of what this does]
# ⚠️ REQUIRES ADMIN (only if true)
cd C:\Github\LLM_fine-tuning
.\path\to\script.ps1 -Flag value
```

### PowerShell Regex Syntax
PowerShell uses .NET regex, NOT bash/grep regex:

| Pattern | Bash/grep | PowerShell |
|---------|-----------|------------|
| OR | `\|` | `|` (no escape) |
| Word boundary | `\b` | `\b` (same) |
| Escape special | `\.` | `\.` (same) |
| Case sensitivity | Case-sensitive default | Case-insensitive default |
```powershell
# ❌ WRONG - bash-style escaped OR
Select-String -Pattern "argparse\|add_argument"

# ✅ CORRECT - .NET regex OR
Select-String -Pattern "argparse|add_argument"

# ❌ WRONG - grep doesn't exist in PowerShell
Get-Content file.txt | grep "pattern"

# ✅ CORRECT - PowerShell equivalent
Get-Content file.txt | Select-String -Pattern "pattern"
```

---

## 🔴 FILE ENCODING

Every `open()` call MUST specify `encoding='utf-8'`:

```python
# ✅ REQUIRED
with open('file.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(content)

# ❌ WILL FAIL ON WINDOWS
with open('file.json', 'r') as f:  # Missing encoding
    data = json.load(f)
```

**This applies to**: `open()`, `Path.read_text()`, `Path.write_text()`, `json.load()`, `yaml.safe_load()`, CSV readers, etc.

---

## 🔴 TRAINING OBJECTIVE: MAXIMUM QUALITY

**Prime Directive**: Create a PERFECT fine-tuned model. Cost, time, and complexity are NOT constraints.

### Available Resources (FULLY UTILIZE)
| Resource | Specification | Utilization Target |
|----------|---------------|-------------------|
| GPU | NVIDIA RTX A6000 | Fill 48GB VRAM |
| RAM | 128GB | Use for large datasets |
| CPU | AMD Threadripper 7960X (24c/48t) | Parallelize preprocessing |

### Decision Framework
When choosing between options:
```
Quality Option A (10x cost, 2x benefit) → ✅ CHOOSE THIS
Efficient Option B (1x cost, 1x benefit) → ❌ REJECT
```

### Training Defaults (START HERE, not "work up to")
| Parameter | Minimum Value | Rationale |
|-----------|---------------|-----------|
| LoRA rank | 256+ | Higher capacity |
| Context length | 8192+ | Full context utilization |
| Batch size | Fill VRAM | Maximum throughput |
| Epochs | 3-5 minimum | Thorough learning |

### Checkpoint Requirements (MANDATORY)
Every training config MUST include:
```yaml
save_steps: 250
save_total_limit: 5
save_only_model: false
resume_from_checkpoint: false  # Set path to resume
```

---

## 🟡 RECOMMENDATION FORMAT

When presenting options, use this structure:

```markdown
## Objective Reminder
[State what we're optimizing for]

## Options (Ranked by Expected Quality)

### 🥇 Option 1: [Best Quality]
- **Quality**: [Expected outcome]
- **Resource Usage**: GPU: X%, RAM: Y GB, Time: Z hrs
- **Tradeoff**: [What this costs]

### 🥈 Option 2: [Second Best]
...

## Recommendation
I recommend **Option [N]** because [quality rationale].
```

**Never**:
- Present conservative/cheap options as defaults
- Omit complex options because they're harder
- Be neutral when one option is clearly better

---

## 🟡 CODE STYLE

### Standards
- Google Python Style Guide
- Ruff for linting/formatting
- Line length: 119 characters
- Quote style: double quotes
- Docstrings: Google style

### Import Order
```python
# Standard library
import os
import sys

# Third-party (2 blank lines after)
import torch
import transformers
from accelerate import Accelerator

# First-party (2 blank lines after)
from llamafactory.data import DataProcessor
from llamafactory.model import load_model
```

### License Header (Required)
All source files must include Apache 2.0 header.

---

## 🟢 PROJECT STRUCTURE REFERENCE

### v0 Architecture (Default)
```
src/llamafactory/
├── api/          # OpenAI-style API
├── chat/         # Chat interface
├── data/         # Dataset handling
├── eval/         # Evaluation utilities
├── extras/       # Helpers
├── hparams/      # Hyperparameter dataclasses
├── model/        # Model loading/patching
├── train/        # Training pipelines (SFT, DPO, PPO, RM, PT, KTO, ORPO)
└── webui/        # Gradio interface
```

### v1 Architecture (USE_V1=1)
```
src/llamafactory/v1/
├── trainers/     # Training implementations
├── core/         # Core utilities
├── accelerator/  # Distributed training
├── plugins/      # Pluggable components
├── config/       # Configuration management
└── utils/        # Utilities
```

### Entry Points
```bash
llamafactory-cli train --config examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli webui
llamafactory-cli api
llamafactory-cli chat --model_name_or_path MODEL_PATH
```

---

## 🟢 QUALITY COMMANDS

```powershell
# Before committing
make style      # Auto-fix formatting
make quality    # Run linters
make test       # Run pytest suite
make commit     # All pre-commit hooks
make license    # Check license headers
```

---

## QUICK REFERENCE CHECKLIST

Before suggesting ANY command:
- [ ] Verified file/path exists with `view` tool
- [ ] Checked script interface (argparse flags)
- [ ] Used PowerShell syntax (not bash)
- [ ] Included `encoding='utf-8'` in file operations
- [ ] Provided absolute paths
- [ ] Stated if admin privileges required

Before recommending training config:
- [ ] Included checkpoint settings
- [ ] Maximized for quality (not efficiency)
- [ ] Utilized full GPU memory
- [ ] Used high LoRA rank (256+)

Before presenting options:
- [ ] Ranked by quality (best first)
- [ ] Made clear recommendation
- [ ] Showed resource utilization
