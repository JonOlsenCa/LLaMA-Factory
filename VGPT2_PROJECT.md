# VGPT2 Project Documentation

**Project:** VGPT2 (Virtual GPT 2) - AP Wizard Database Expert
**Status:** Training Complete, Initial Testing Done
**Last Updated:** 2025-12-29

---

> ⚠️ **CRITICAL: VS Code Terminal Bug**
>
> VS Code's integrated terminal randomly sends `Ctrl+C` interrupts which kills long-running scripts (training, testing, inference). This is a known issue.
>
> **MANDATORY:** Always run scripts in an external terminal window using:
> ```powershell
> .\scripts\run_external.ps1 -Script "test_vgpt2.py"
> .\scripts\start_training.ps1
> ```
>
> If processes get orphaned, clean them up with:
> ```powershell
> .\scripts\kill_orphans.ps1
> ```

---

## Executive Summary

VGPT2 is a LoRA fine-tuned version of Qwen2.5-7B-Instruct trained on 23,742 records of AP Wizard database documentation. Training completed successfully after 6.5 hours. Initial testing shows the model has learned database schema knowledge but exhibits some hallucination issues (inventing table names that don't exist).

---

## 1. What Was Done

### Training Completed
- **Date:** 2025-12-29 07:27:30
- **Duration:** 6 hours 35 minutes
- **Steps:** 7,050 total (resumed from checkpoint-1600)
- **Final Train Loss:** 0.2454
- **Final Eval Loss:** 1.171

### Model Details
| Component | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-7B-Instruct (7.9B params) |
| Fine-tuning Method | LoRA |
| LoRA Rank | 128 |
| LoRA Alpha | 256 |
| Trainable Params | 322,961,408 (4.07%) |
| Adapter Size | 1.23 GB |

### Training Data
- **Dataset:** `vgpt2_v2` - 23,742 records from 14 sources
- **Content:** Database schemas, column definitions, SQL examples, business logic
- **Format:** Alpaca-style (instruction/input/output)
- **Validation Split:** 5% (~1,187 samples)

### Hardware Used
- GPU: NVIDIA RTX A6000 (48GB VRAM)
- RAM: 128GB
- CPU: AMD Threadripper 7960X

---

## 2. How It Was Done

### Training Configuration
File: `automation/configs/vgpt2_lora_sft.yaml`

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_target: all
dataset: vgpt2_v2
template: qwen
cutoff_len: 4096
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 5.0
lr_scheduler_type: cosine
bf16: true
```

### Training Command
```powershell
cd C:\Github\LLM_fine-tuning
.\venv\Scripts\activate
llamafactory-cli train automation/configs/vgpt2_lora_sft.yaml
```

### Resume from Checkpoint
Training was interrupted and resumed from checkpoint-1600:
```yaml
resume_from_checkpoint: true
```

---

## 3. Testing Methodology

### Test Script
File: `scripts/test_vgpt2.py`

```python
# Loads base model + LoRA adapter
# Runs test questions through the model
# Uses: temperature=0.7, top_p=0.8, max_new_tokens=256
```

### How to Run Tests
```powershell
cd C:\Github\LLM_fine-tuning
.\venv\Scripts\activate
python scripts/test_vgpt2.py
```

---

## 4. Test Results

### Test Run 1: Basic Schema Questions (PASSED ✅)

| Question | Response | Accuracy |
|----------|----------|----------|
| "What columns are in the Batch.Document view?" | Listed 21 columns correctly | ✅ Correct |
| "Describe key columns in Analytics.vDataSyncTracking" | Tracking_Id, Extract_Date, Extract_Run_Id with types | ✅ Correct |
| "What is the data type of DocumentId in Batch.Document?" | uniqueidentifier | ✅ Correct |

### Test Run 2: SQL Generation (MIXED ⚠️)

| Question | Response | Accuracy |
|----------|----------|----------|
| "How do I join the Invoice and Vendor tables?" | Suggested `FROM Invoice INNER JOIN Vendor ON Invoice.VendorGroup = Vendor.VendorGroup` | ❌ **WRONG** - "Invoice" is not a real SQL object |

### Key Finding: Hallucination Issue
The model invented a table name "Invoice" that doesn't exist in the actual database. The correct object would be a specific view like `vrvAP_MVAllInvoices` or a table with the proper schema prefix.

**Root Cause Hypothesis:** 
- Training data may have used simplified/generic names in some Q&A pairs
- Model may be generalizing from patterns rather than exact object names
- Need more explicit training on "these are the ONLY valid object names"

---

## 5. Output Files

Location: `saves/vgpt2_v2_lora_sft/`

| File | Size | Purpose |
|------|------|---------|
| adapter_model.safetensors | 1.23 GB | LoRA weights |
| adapter_config.json | <1 KB | LoRA config |
| tokenizer.json | 10.9 MB | Tokenizer |
| training_loss.png | - | Loss curve |
| trainer_state.json | - | Training state |

---

## 6. Dependencies (CRITICAL - Pinned Versions)

These exact versions are required. Upgrading breaks LlamaFactory.

| Package | Required Version | DO NOT USE |
|---------|-----------------|------------|
| Python | 3.12 | 3.13, 3.14 |
| torch | 2.6.0+cu124 | - |
| transformers | 4.57.1 | 4.57.3 |
| peft | 0.17.1 | 0.18.0 |
| accelerate | 1.11.0 | 1.12.0 |

### Install Commands
```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.1 peft==0.17.1 accelerate==1.11.0
pip install -e ".[torch,metrics]"
```

---

## 7. Proposed Next Steps

### Based on Test Results:

#### If Model Accuracy is GOOD (>80% correct):
1. **Expand test suite** - Create 50+ validation questions covering all schema areas
2. **Merge adapter** - Combine LoRA weights with base model for faster inference
3. **Deploy** - Integrate into AP Wizard application via API
4. **Monitor** - Track accuracy in production, collect failure cases for retraining

#### If Model Accuracy is POOR (<80% correct):
1. **Analyze failures** - Categorize errors (hallucination, wrong columns, wrong types)
2. **Improve training data:**
   - Add negative examples ("Invoice is NOT a valid table name")
   - Add explicit object name lists
   - Increase examples for failed categories
3. **Retrain** - Use improved dataset, possibly with higher LoRA rank (256)
4. **Consider RAG** - Hybrid approach: model + retrieval from actual schema docs

#### Current Status: MIXED RESULTS
- Schema questions: ✅ Working well
- SQL generation: ⚠️ Hallucinating object names

### Recommended Immediate Actions:
1. Run comprehensive test suite (50+ questions)
2. Document all hallucination cases
3. Review training data for problematic patterns
4. Consider adding "valid object name" constraints to training

---

## 8. Proposed Test Questions

### Schema Knowledge Tests
```
1. What columns are in dbo.bAPTH?
2. List all views in the Analytics schema
3. What is the primary key of bAPUI?
4. Describe the bAPVendor table structure
5. What columns link bAPTD to bAPTH?
```

### SQL Generation Tests
```
6. Write SQL to get all unpaid invoices from bAPTH
7. How do I join bAPTH and bAPVendor?
8. Query to find duplicate invoice numbers
9. Get total invoice amount by vendor for 2024
10. Find all invoices approved but not paid
```

### Edge Case Tests
```
11. What is the "Invoice" table? (Should say it doesn't exist)
12. How do I query the UserPreferences table? (May not exist)
13. What's the difference between bAPTH and vrvAP_MVAllInvoices?
```

### Business Logic Tests
```
14. How does the payment approval workflow work?
15. What triggers a duplicate invoice warning?
16. When is an invoice considered "paid"?
```

---

## 9. File Structure

```
C:\Github\LLM_fine-tuning\
├── automation/
│   └── configs/
│       └── vgpt2_lora_sft.yaml      # Training config
├── data/
│   ├── dataset_info.json            # Dataset registry
│   └── vgpt2_training_v2.json       # Training data (23,742 records)
├── saves/
│   └── vgpt2_v2_lora_sft/           # Trained model output
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── ...
├── scripts/
│   ├── test_vgpt2.py                # Test script
│   ├── resource_monitor.py          # GPU/RAM monitor
│   └── maximize_gpu.ps1             # GPU optimization
├── VGPT2_PROJECT.md                 # This file
├── VGPT2_TRAINING_COMPLETE.md       # Training summary
├── RESUME_TRAINING.md               # Resume instructions
├── SETUP.md                         # Setup guide
└── requirements.txt                 # Dependencies
```

---

## 10. Quick Reference Commands

> ⚠️ **ALWAYS use external terminal scripts** - Never run these directly in VS Code!

### Run Test Script (CORRECT WAY)
```powershell
.\scripts\run_external.ps1 -Script "test_vgpt2.py"
```

### Start Training (CORRECT WAY)
```powershell
.\scripts\start_training.ps1
.\scripts\start_training.ps1 -Resume    # Resume from checkpoint
```

### Interactive Chat (CORRECT WAY)
```powershell
.\scripts\run_external.ps1 -Command "llamafactory-cli chat --model_name_or_path Qwen/Qwen2.5-7B-Instruct --adapter_name_or_path saves/vgpt2_v2_lora_sft --template qwen --finetuning_type lora"
```

### Web UI (CORRECT WAY)
```powershell
.\scripts\run_external.ps1 -Command "set GRADIO_SERVER_NAME=127.0.0.1; llamafactory-cli webui"
```

### Monitor Resources
```powershell
.\scripts\run_external.ps1 -Script "resource_monitor.py"
```

### Kill Orphaned Processes
```powershell
.\scripts\kill_orphans.ps1           # Interactive
.\scripts\kill_orphans.ps1 -Force    # Kill all
.\scripts\kill_orphans.ps1 -List     # List only
```

---

## 11. External Terminal Scripts

These scripts are **MANDATORY** for running anything in this project.

| Script | Purpose |
|--------|---------|
| `scripts/run_external.ps1` | Run any script/command in external terminal |
| `scripts/start_training.ps1` | Launch training in external terminal |
| `scripts/kill_orphans.ps1` | Clean up orphaned Python processes |
| `scripts/resource_monitor.py` | Monitor GPU/CPU/RAM usage |
| `scripts/test_vgpt2.py` | Test model with sample questions |

### Why External Terminals?

VS Code's integrated terminal has a bug where it randomly sends `Ctrl+C` (KeyboardInterrupt) signals to running processes. This causes:
- Training to stop mid-run
- Test scripts to crash
- Model loading to fail
- SSL/network operations to timeout

**Symptoms of this bug:**
```
KeyboardInterrupt
(venv) PS C:\Github\LLM_fine-tuning> ^C
```

The `^C` appears even though you didn't press anything. This is VS Code, not you.

