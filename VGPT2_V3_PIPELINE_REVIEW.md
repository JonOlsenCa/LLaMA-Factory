# VGPT2 v3 Training Pipeline - Critical Assessment

**Date:** 2025-12-29  
**Reviewer:** Augment Agent  
**Status:** Pre-Training Review

---

## Executive Summary

You've built a well-structured, comprehensive training pipeline. The multi-stage approach (SFT → DPO → KTO) is sound, and you've correctly identified the key weaknesses of v2 (hallucination, missing negative examples). However, there are **significant gaps between your plan and reality** that need to be addressed before training begins.

---

## 1. What's Strong ✅

### A. Architecture & Methodology
- **Three-stage training pipeline (SFT → DPO → KTO)** is the correct modern approach for alignment
- **LLaMA-Factory integration** is well-configured with proper Qwen2.5 template
- **Hardware utilization** is good: LoRA rank 256, gradient checkpointing, appropriate batch sizes for 48GB VRAM
- **Dataset format registrations** in `dataset_info.json` are correctly configured for all three stages

### B. Anti-Hallucination Focus
- Negative examples generator explicitly teaches model to refuse fake tables
- DPO pairs include hallucination rejection patterns
- Test suite includes dedicated hallucination tests

### C. Validation Pipeline
- Well-designed `TestSuite` class with clear categories
- Includes expected keywords, forbidden keywords, and refusal detection
- Both quick (50) and full (500) test suites available

### D. Training Configs
- Reasonable hyperparameters (LR 1.5e-4 for SFT, 5e-6 for DPO/KTO)
- Proper learning rate decay (cosine)
- Validation holdout at each stage (2%, 5%, 10%)

---

## 2. Critical Gaps & Weaknesses ⚠️

### A. **MASSIVE Data Shortfall** (CRITICAL)

| Dataset | Plan Target | Actual Generated | Shortfall |
|---------|-------------|------------------|-----------|
| SFT | 100,000 | **45,467** | **54.5% missing** |
| Negatives | 2,000 | **431** | **78.5% missing** |
| DPO Pairs | 5,000 | **1,428** | **71.4% missing** |
| KTO | 3,000 | **1,118** | **62.7% missing** |

**This is your #1 risk.** The plan promises 100K+ records but you only have 45K. The negative examples are critically undersized at just 431.

### B. **KTO Format Mismatch** (HIGH)

Your KTO generator outputs:
```python
"label": self.label  # boolean True/False
```

But LLaMA-Factory expects `kto_tag` with string values `"true"` or `"false"`. Your `dataset_info.json` maps `kto_tag` → `label`, but the actual values are Python booleans, not strings.

**Fix required in `generate_kto_data.py`:**
```python
def to_dict(self) -> Dict:
    return {
        "instruction": self.instruction,
        "input": self.input,
        "output": self.output,
        "label": "true" if self.label else "false"  # String, not bool!
    }
```

### C. **Negative Examples Not Integrated into SFT** (HIGH)

You have 431 negative examples in `vgpt2_v3_negatives.json`, but:
- Your SFT config only uses `vgpt2_v3_sft` dataset
- Negatives are a **separate file** not merged into SFT data
- The generator in `generate_training_data.py` doesn't call `generate_negative_examples.py`

**The negative examples won't be used during SFT training as currently configured.**

### D. **DPO/KTO Data Overlap Risk** (MEDIUM)

Your DPO and KTO generators create similar examples from the same schema data:
- Both generate NOLOCK preference patterns
- Both generate hallucination examples
- Both use the same random sampling approach

This overlap may reduce the effectiveness of the progressive refinement approach.

### E. **Validation Pipeline Silent Fallback** (MEDIUM)

In `run_validation.py`, if the model path doesn't exist, it silently falls back to base model - you could validate the wrong model without realizing it.

### F. **No SQL Syntax Validation** (MEDIUM)

The plan mentions "SQL Validation Checks" with SQL Server parser, column verification, etc. But `run_validation.py` only does keyword matching. There's no actual SQL parsing.

---

## 3. Missing Elements

| Element | Status | Impact |
|---------|--------|--------|
| Dataset Deduplication | Missing | Duplicate entries reduce effective training |
| Data Quality Validation Script | Missing | Can't verify data before training |
| Checkpoint Comparison Tools | Missing | Can't compare SFT vs DPO systematically |
| Early Stopping Criteria | Missing | Risk of overfitting |
| Training Orchestration Script (`run_pipeline.ps1`) | Missing | Manual stage execution required |
| SQL Syntax Parser | Missing | Can't validate SQL correctness |

---

## 4. Recommendations

### Immediate (Before Training)

#### 1. Fix the KTO format
Update `scripts/vgpt2_v3/generate_kto_data.py` to output string labels:
```python
"label": "true" if self.label else "false"
```

#### 2. Merge negatives into SFT data
Option A - Create merged file:
```bash
python -c "
import json
sft = json.load(open('data/vgpt2_v3_sft.json', encoding='utf-8'))
neg = json.load(open('data/vgpt2_v3_negatives.json', encoding='utf-8'))
merged = sft + neg
json.dump(merged, open('data/vgpt2_v3_sft_merged.json', 'w', encoding='utf-8'), indent=2)
print(f'Merged: {len(merged)} records')
"
```

Option B - Use multi-dataset in config:
```yaml
dataset: vgpt2_v3_sft,vgpt2_v3_negatives
```

#### 3. Add fail-fast validation check
In `run_validation.py`:
```python
if not self.model_path.exists():
    raise ValueError(f"Model adapter not found at {self.model_path}")
```

#### 4. Run data validation before training
```python
import json
for file in ['vgpt2_v3_sft.json', 'vgpt2_v3_dpo.json', 'vgpt2_v3_kto.json']:
    data = json.load(open(f'data/{file}', encoding='utf-8'))
    print(f"{file}: {len(data)} records")
    print(f"  Fields: {list(data[0].keys())}")
```

### Short-term (Improve Data Quality)

#### 5. Expand data generators to meet targets
- Remove/increase `max_records` limits in generators
- Add more generation patterns
- Mine more from VGPT2 sources (Tables/, Functions/, Triggers/)

#### 6. Add deduplication
```python
seen = set()
unique = []
for r in records:
    key = r['instruction'].lower().strip()
    if key not in seen:
        seen.add(key)
        unique.append(r)
```

#### 7. Create separate test data file
Move test questions to `test_suite.json` for versioning and extension without code changes.

---

## 5. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data shortfall reduces accuracy | HIGH | HIGH | Expand generators, mine more sources |
| KTO format fails | HIGH | HIGH | Fix to string format now |
| Negatives unused | HIGH | MEDIUM | Merge into SFT or multi-dataset |
| DPO causes regression | MEDIUM | HIGH | Checkpoint after SFT, compare |
| Hallucination persists | MEDIUM | HIGH | More negative examples, more DPO pairs |
| Training OOM | LOW | MEDIUM | Already have checkpointing; reduce batch if needed |
| Validation tests wrong model | MEDIUM | HIGH | Add path existence check |

---

## 6. Suggested Order of Operations

```
Day 1: Data Preparation
├── [x] Generate SFT data (45K done, need ~55K more)
├── [ ] Fix KTO format (string not bool)
├── [ ] Merge negatives into SFT
├── [ ] Run deduplication
├── [ ] Create data validation script
└── [ ] Validate all datasets

Day 2: Pre-Training Validation
├── [ ] Run validation with BASE model (establish baseline)
├── [ ] Verify dataset_info.json is correct
├── [ ] Test-load one batch from each dataset
└── [ ] Document baseline metrics

Day 3-4: Stage 1 (SFT)
├── [ ] Run SFT training (~8-12 hrs)
├── [ ] Monitor loss curves
├── [ ] Run validation on SFT model
├── [ ] Compare to baseline (should see ~80%+ improvement)
└── [ ] CHECKPOINT DECISION: Continue or iterate?

Day 5: Stage 2 (DPO)
├── [ ] Run DPO training (~4-6 hrs)
├── [ ] Validate DPO model
├── [ ] Compare to SFT (should maintain accuracy, reduce hallucination)
└── [ ] CHECKPOINT DECISION

Day 6: Stage 3 (KTO)
├── [ ] Run KTO training (~2-4 hrs)
├── [ ] Final validation
├── [ ] Full regression test
└── [ ] Document final metrics

Day 7: Production Prep
├── [ ] Export final model
├── [ ] Integration testing
└── [ ] Deploy
```

---

## 7. Quick Wins (Do These Now)

### 1. Create a merged SFT dataset
```bash
python -c "
import json
sft = json.load(open('data/vgpt2_v3_sft.json', encoding='utf-8'))
neg = json.load(open('data/vgpt2_v3_negatives.json', encoding='utf-8'))
merged = sft + neg
json.dump(merged, open('data/vgpt2_v3_sft_merged.json', 'w', encoding='utf-8'), indent=2)
print(f'Merged: {len(merged)} records')
"
```

### 2. Update stage1_sft.yaml
```yaml
dataset: vgpt2_v3_sft_merged
```

### 3. Verify KTO data format
```bash
python -c "
import json
data = json.load(open('data/vgpt2_v3_kto.json', encoding='utf-8'))
sample = data[0]
print(f'Label type: {type(sample.get(\"label\"))}')
print(f'Label value: {sample.get(\"label\")}')
"
```

---

## 8. Data Quality Summary

### Current State

| File | Records | Status |
|------|---------|--------|
| `vgpt2_v3_sft.json` | 45,467 | ⚠️ Below target (100K) |
| `vgpt2_v3_negatives.json` | 431 | ⚠️ Below target (2K), not merged |
| `vgpt2_v3_dpo.json` | 1,428 | ⚠️ Below target (5K) |
| `vgpt2_v3_kto.json` | 1,118 | ⚠️ Below target (3K), format issue |

### Minimum Viable Targets

For a reasonable chance at 90%+ accuracy:
- **SFT:** 75,000+ records (currently at 60%)
- **Negatives:** 1,500+ records (currently at 29%)
- **DPO:** 3,000+ pairs (currently at 48%)
- **KTO:** 2,000+ examples (currently at 56%)

---

## 9. Conclusion

Your pipeline is **architecturally sound** but has **critical data gaps**. The 45K SFT records may still achieve significant improvement over v2's 64% accuracy, but you're unlikely to hit the 90%+ target without:

1. More training data (aim for 75K+ minimum)
2. More negative examples (aim for 1,500+ minimum)
3. Proper integration of anti-hallucination data
4. Fixed KTO format

**Recommended action:** Spend 1-2 more days on data generation before starting training. The ROI on better data far exceeds the cost of longer training or multiple iteration cycles.

---

## 10. Files Reviewed

| File | Purpose |
|------|---------|
| `docs/VGPT2_V3_MASTER_PLAN.md` | Comprehensive plan document |
| `automation/configs/vgpt2_v3/stage1_sft.yaml` | SFT training config |
| `automation/configs/vgpt2_v3/stage2_dpo.yaml` | DPO training config |
| `automation/configs/vgpt2_v3/stage3_kto.yaml` | KTO training config |
| `scripts/vgpt2_v3/generate_training_data.py` | Main data generator |
| `scripts/vgpt2_v3/generate_dpo_pairs.py` | DPO pairs generator |
| `scripts/vgpt2_v3/generate_kto_data.py` | KTO data generator |
| `scripts/vgpt2_v3/generate_negative_examples.py` | Anti-hallucination data |
| `scripts/vgpt2_v3/run_validation.py` | Validation pipeline |
| `data/dataset_info.json` | Dataset registrations |

---

*Review completed: 2025-12-29*

