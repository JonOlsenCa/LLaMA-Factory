# VGPT2 v3 Master Training Plan

**Version:** 3.0
**Date:** 2025-12-29
**Status:** Implementation Ready
**Goal:** Production-grade Viewpoint Vista SQL Expert (>90% accuracy)

---

## Executive Summary

VGPT2 v2 achieved 64% accuracy with significant hallucination issues (67% on edge cases). This plan leverages the full capabilities of the available hardware and LLaMA-Factory framework to build a production-ready system.

### Key Changes from v2

| Aspect | v2 (Current) | v3 (This Plan) |
|--------|--------------|----------------|
| Training Data | 23,742 records | **100,000+ records** |
| Negative Examples | 0 | **2,000+** |
| Training Method | SFT only | **SFT → DPO → KTO** |
| Model Options | 7B LoRA only | **7B Full / 32B LoRA / Ensemble** |
| Validation | 53 manual tests | **500+ automated tests** |
| SQL Validation | None | **Syntax + Schema verification** |

---

## Hardware Resources

| Component | Specification | Utilization Plan |
|-----------|---------------|------------------|
| GPU | NVIDIA RTX A6000 (48GB VRAM) | Full utilization with gradient checkpointing |
| CPU | AMD Threadripper 7960X (48 cores) | 16-32 workers for data preprocessing |
| RAM | 128GB DDR5 | Model offloading, large batch caching |
| Storage | SSD | Checkpoint management |

---

## Phase 1: Data Expansion

### 1.1 Current Data Sources (Used)

| Source | Records | Status |
|--------|---------|--------|
| Schema Metadata | 7,898 | ✅ Used |
| SP Documentation | 7,040 | ✅ Used |
| View Documentation | 2,350 | ✅ Used |
| DDFI Forms | 1,692 | ✅ Used |
| Crystal Report SQL | 1,217 | ✅ Used |
| JOIN Patterns | 1,033 | ✅ Used |
| Experts V2 | 405 | ✅ Used |
| **Total v2** | **23,742** | |

### 1.2 Untapped Data Sources

| Source | Location | Est. Records | Priority |
|--------|----------|--------------|----------|
| Full Column Details | `_Metadata/columns.json` | 40,000+ | P1 |
| Table Documentation | `Tables/` | 8,000+ | P1 |
| Function Documentation | `Functions/` | 5,000+ | P2 |
| Trigger Documentation | `Triggers/` | 5,000+ | P3 |
| Full DDFI Fields | `_Metadata/DDFI.json` | 10,000+ | P2 |
| All Crystal Reports | `Crystal_Reports_SQL/` | 2,400+ | P1 |
| Vista KB Articles | External | 5,000+ | P3 |

### 1.3 Synthetic Data Generation

| Type | Method | Target Records |
|------|--------|----------------|
| Negative Examples | Rule-based generation | 2,000 |
| Complex SQL Scenarios | Template expansion | 5,000 |
| Error Correction Pairs | Pattern-based | 1,000 |
| Business Logic Q&A | Documentation parsing | 3,000 |

### 1.4 Target Data Distribution

| Category | v2 Count | v3 Target | Percentage |
|----------|----------|-----------|------------|
| Schema Knowledge | 7,898 | 25,000 | 25% |
| SQL Generation | 2,018 | 20,000 | 20% |
| SP/View Documentation | 9,390 | 15,000 | 15% |
| Negative Examples | 0 | 5,000 | 5% |
| Error Correction | 8 | 5,000 | 5% |
| JOIN Patterns | 1,033 | 8,000 | 8% |
| Business Context | 1,696 | 10,000 | 10% |
| Crystal Report SQL | 1,217 | 5,000 | 5% |
| Canonical Rules | 26 | 2,000 | 2% |
| Query Optimization | 5 | 2,000 | 2% |
| Naming Conventions | 26 | 1,500 | 1.5% |
| Heuristics/Workflows | 8 | 1,500 | 1.5% |
| **TOTAL** | **23,742** | **100,000** | 100% |

---

## Phase 2: Multi-Stage Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Primary knowledge injection stage.

```
Dataset: vgpt2_v3_sft (100K records)
     ↓
Base Model → LoRA/Full Fine-tune → SFT Model
     ↓
Checkpoint: saves/vgpt2_v3_sft/
```

**Configuration:** `automation/configs/vgpt2_v3/stage1_sft.yaml`

### Stage 2: Direct Preference Optimization (DPO)

Teaches model to prefer correct SQL over incorrect SQL.

```
Dataset: vgpt2_v3_dpo (5K preference pairs)
     ↓
SFT Model → DPO Training → DPO Model
     ↓
Checkpoint: saves/vgpt2_v3_dpo/
```

**Preference Pair Format:**
```json
{
  "instruction": "Write SQL to find unpaid invoices",
  "chosen": "SELECT * FROM APTH WITH (NOLOCK) WHERE APCo = @APCo AND Status = 0",
  "rejected": "SELECT * FROM Invoice WHERE Paid = 0"
}
```

**Configuration:** `automation/configs/vgpt2_v3/stage2_dpo.yaml`

### Stage 3: Kahneman-Tversky Optimization (KTO)

Binary feedback refinement for edge cases.

```
Dataset: vgpt2_v3_kto (3K examples with binary labels)
     ↓
DPO Model → KTO Training → Final Model
     ↓
Checkpoint: saves/vgpt2_v3_final/
```

**Configuration:** `automation/configs/vgpt2_v3/stage3_kto.yaml`

---

## Phase 3: Model Architecture Options

### Option A: Qwen2.5-32B-Instruct with QLoRA (Recommended)

- **Pros:** 4x more parameters, dramatically better reasoning
- **Cons:** Slower inference, requires quantization
- **VRAM:** ~40GB with 4-bit quantization + LoRA

### Option B: Qwen2.5-7B Full Fine-Tune

- **Pros:** All parameters updated, deep learning
- **Cons:** Longer training (~24h), needs DeepSpeed
- **VRAM:** ~45GB with ZeRO-3 offloading

### Option C: Qwen2.5-7B LoRA with Higher Rank

- **Pros:** Fastest training, good baseline
- **Cons:** Limited by LoRA capacity
- **VRAM:** ~40GB with rank 256

### Recommended Approach

Run Option C first as baseline, then Option A for production.

---

## Phase 4: Validation Pipeline

### 4.1 Automated Test Suite

| Category | Questions | Pass Criteria |
|----------|-----------|---------------|
| Schema Knowledge | 100 | >95% correct |
| SQL Generation | 150 | >90% valid syntax |
| Hallucination Tests | 100 | <5% hallucination |
| JOIN Patterns | 50 | >90% correct |
| Error Correction | 50 | >85% proper fix |
| Business Logic | 50 | >90% correct |
| **Total** | **500** | |

### 4.2 SQL Validation Checks

1. **Syntax Validation:** Parse with SQL Server parser
2. **Column Verification:** Check against `columns.json`
3. **Table Verification:** Check against `_Viewpoint_ALL_Views_Tables_Complete.json`
4. **JOIN Validation:** Verify against `foreign_keys.json`
5. **Convention Compliance:** WITH (NOLOCK), no aliases, proper case

### 4.3 Regression Testing

After each training stage:
```bash
python scripts/vgpt2_v3/run_validation.py --model saves/vgpt2_v3_sft --suite full
```

---

## Phase 5: Production Deployment

### 5.1 Hybrid RAG + Fine-Tuned Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Query                         │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│              Query Classifier                        │
│  (Schema lookup? SQL generation? Explanation?)       │
└─────────────────────┬───────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
┌─────────────┐ ┌──────────┐ ┌──────────────┐
│ Schema RAG  │ │ VGPT2 v3 │ │ Explanation  │
│ (columns,   │ │ (SQL Gen)│ │ (business    │
│  tables)    │ │          │ │  context)    │
└──────┬──────┘ └────┬─────┘ └──────┬───────┘
       │             │              │
       └─────────────┼──────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│              SQL Validator                           │
│  (Syntax check, column verify, JOIN validate)        │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│              Final Response                          │
└─────────────────────────────────────────────────────┘
```

### 5.2 Inference Configuration

```yaml
# Optimized inference settings
temperature: 0.3          # Lower for consistent SQL
top_p: 0.9
max_new_tokens: 1024
repetition_penalty: 1.1
```

---

## Directory Structure

```
automation/
├── configs/
│   └── vgpt2_v3/
│       ├── stage1_sft.yaml         # SFT training config
│       ├── stage2_dpo.yaml         # DPO training config
│       ├── stage3_kto.yaml         # KTO training config
│       ├── inference.yaml          # Inference config
│       └── model_options/
│           ├── qwen32b_qlora.yaml  # 32B option
│           ├── qwen7b_full.yaml    # Full fine-tune option
│           └── qwen7b_lora.yaml    # Enhanced LoRA option

scripts/
└── vgpt2_v3/
    ├── generate_training_data.py   # Expanded data generator
    ├── generate_negative_examples.py
    ├── generate_dpo_pairs.py
    ├── generate_kto_examples.py
    ├── run_validation.py           # Automated test suite
    ├── validate_sql.py             # SQL validation utilities
    ├── run_pipeline.ps1            # Full training orchestration
    └── utils/
        ├── schema_loader.py
        ├── sql_parser.py
        └── test_questions.json

data/
├── vgpt2_v3_sft.json              # 100K SFT records
├── vgpt2_v3_dpo.json              # 5K preference pairs
├── vgpt2_v3_kto.json              # 3K binary feedback
└── validation/
    ├── test_suite_500.json
    └── results/

saves/
└── vgpt2_v3/
    ├── sft/                        # Stage 1 checkpoints
    ├── dpo/                        # Stage 2 checkpoints
    └── final/                      # Stage 3 final model

docs/
├── VGPT2_V3_MASTER_PLAN.md        # This document
├── VGPT2_V3_DATA_SPEC.md          # Data format specifications
└── VGPT2_V3_VALIDATION_SPEC.md   # Validation criteria
```

---

## Execution Timeline

| Day | Phase | Activities |
|-----|-------|------------|
| 1 | Data Prep | Run expanded data generators, validate output |
| 2 | Data Prep | Generate negative examples, DPO pairs, KTO data |
| 3 | Stage 1 | SFT training (~8-12 hours) |
| 4 | Validation | Test SFT model, analyze failures |
| 5 | Stage 2 | DPO training (~4-6 hours) |
| 6 | Stage 3 | KTO training (~2-4 hours), final validation |
| 7 | Deploy | Production integration, monitoring setup |

---

## Success Criteria

| Metric | v2 Baseline | v3 Target | Stretch Goal |
|--------|-------------|-----------|--------------|
| Overall Accuracy | 64% | 90% | 95% |
| Hallucination Rate | 67% | <5% | <2% |
| SQL Syntax Valid | 87% | 99% | 100% |
| Schema Knowledge | 67% | 95% | 98% |
| Complex JOINs | Unknown | 85% | 92% |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting on expanded data | Early stopping, validation monitoring |
| 32B model doesn't fit in VRAM | Fall back to 7B full fine-tune |
| DPO causes regression | Checkpoint after SFT, A/B compare |
| Training crashes | Checkpoint every 200 steps, resume support |
| Data quality issues | Validation pipeline before training |

---

## Commands Quick Reference

```powershell
# Generate expanded training data
python scripts/vgpt2_v3/generate_training_data.py --output data/vgpt2_v3_sft.json

# Stage 1: SFT Training
.\scripts\run_external.ps1 -Command "llamafactory-cli train automation/configs/vgpt2_v3/stage1_sft.yaml"

# Stage 2: DPO Training
.\scripts\run_external.ps1 -Command "llamafactory-cli train automation/configs/vgpt2_v3/stage2_dpo.yaml"

# Stage 3: KTO Training
.\scripts\run_external.ps1 -Command "llamafactory-cli train automation/configs/vgpt2_v3/stage3_kto.yaml"

# Run validation suite
python scripts/vgpt2_v3/run_validation.py --model saves/vgpt2_v3/final --suite full

# Interactive testing
.\scripts\run_external.ps1 -Command "llamafactory-cli chat --model_name_or_path Qwen/Qwen2.5-7B-Instruct --adapter_name_or_path saves/vgpt2_v3/final --template qwen"
```

---

## Appendix A: Data Format Specifications

### SFT Format (Alpaca)
```json
{
  "instruction": "User's question or request",
  "input": "Optional context (can be empty)",
  "output": "Expected response"
}
```

### DPO Format
```json
{
  "instruction": "User's question",
  "input": "",
  "chosen": "Correct/preferred response",
  "rejected": "Incorrect/rejected response"
}
```

### KTO Format
```json
{
  "instruction": "User's question",
  "input": "",
  "output": "Model's response",
  "kto_tag": "true"  // or "false"
}
```

---

## Appendix B: Negative Example Categories

1. **Non-existent Tables:** Invoice, Customer, Orders, Products, Users
2. **Wrong Column Names:** JobNumber (should be Job), InvoiceID (should be APTrans)
3. **Invalid SQL Patterns:** Missing WITH (NOLOCK), using table aliases
4. **Wrong JOINs:** Missing company columns, wrong composite keys
5. **Case Errors:** apco instead of APCo, APTH instead of APTH

---

*Document Version: 1.0 | Last Updated: 2025-12-29*
