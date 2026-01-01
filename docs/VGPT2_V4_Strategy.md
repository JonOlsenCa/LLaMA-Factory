# VGPT2 V4 Improvement Strategy

## Executive Summary

**Current State (V3 SFT):**
- Overall Complex Test Score: **39.1%**
- Major Gaps: Complex SQL (35.6%), Business Logic (34.1%), Hallucination (10%)
- Strength: Cross-Module Joins (72.6%)

**Target (V4):**
- Minimum 75% on all categories
- 90%+ hallucination detection (critical for production use)

**Estimated Effort:**
- **NOT 100 iterations** - targeted approach with ~500-1000 high-quality samples
- 2-3 training iterations with validation between each

---

## The "Good Enough" Strategy

Instead of brute-force iteration, we use **targeted gap-filling**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    V4 IMPROVEMENT PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. GAP ANALYSIS        2. TARGETED DATA       3. TRAIN & VALIDATE │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     │
│  │ Run V3 on     │     │ Generate      │     │ Train V4      │     │
│  │ hard questions├────►│ training data ├────►│ LoRA adapter  │     │
│  │ Score each    │     │ for gaps only │     │ Merge if good │     │
│  └───────────────┘     └───────────────┘     └───────┬───────┘     │
│         ▲                                            │              │
│         │                                            │              │
│         └────────────────────────────────────────────┘              │
│                    REPEAT UNTIL 75%+ ALL CATEGORIES                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Identified Gaps & Solutions

### 1. Complex SQL (35.6% → Target 75%)

| Gap | Missing Elements | Training Focus |
|-----|-----------------|----------------|
| AR Aging Buckets | CASE WHEN, 30/60/90 buckets | 20 aging query examples |
| Job Cost Estimates | JCJP, JCCH, JCCP, JCCT joins | 30 JC multi-table examples |
| Retainage vs Non-Retainage | APHD, RetHoldCode lookup | 15 hold status queries |
| OnCost Reconciliation | ocApplyMth/Trans/Line | 10 oncost linkage examples |
| CTEs and Window Functions | WITH, OVER, PARTITION BY | 20 advanced SQL examples |

**Training Samples Needed: ~95**

### 2. Business Logic (34.1% → Target 75%)

| Gap | Missing Knowledge | Training Focus |
|-----|-------------------|----------------|
| Retainage Calculations | MaxRetgPct, CurCost vs OrigCost | 10 retainage explanations |
| SLWI Fields | WCRetAmt, SMRetAmt meaning | 15 subcontract field explanations |
| Cost Projections | JCPR, JCPD, projection workflow | 10 projection explanations |
| Duplicate Detection | APCO settings, DupInvChk | 10 AP workflow examples |

**Training Samples Needed: ~45**

### 3. Hallucination Detection (10% → Target 90%)

| Gap | Issue | Training Focus |
|-----|-------|----------------|
| Fake Tables | Accepts Invoice, Payments, etc. | 50 "does not exist" examples |
| Fake Columns | Accepts PaymentStatus, etc. | 30 column validation examples |
| Alternative Suggestions | Doesn't suggest correct tables | 20 redirection examples |

**Training Samples Needed: ~100**

### 4. Cross-Module Joins (72.6% → Target 85%)

| Gap | Missing Path | Training Focus |
|-----|--------------|----------------|
| SLWI→APTD | Missing APTL intermediate | 10 3-table join examples |
| GL Audit Trail | Source column usage | 10 GL tracing examples |
| PR→JC→GL | Full audit path | 10 cross-module examples |

**Training Samples Needed: ~30**

---

## Total Training Data Requirements

| Category | Current Score | Target | Samples Needed |
|----------|---------------|--------|----------------|
| Complex SQL | 35.6% | 75% | 95 |
| Business Logic | 34.1% | 75% | 45 |
| Hallucination | 10% | 90% | 100 |
| Cross-Module | 72.6% | 85% | 30 |
| **TOTAL** | **39.1%** | **80%** | **~270** |

With 3x augmentation (paraphrasing), we get ~800 training samples.

---

## Automated Improvement Loop

### Scripts Created

1. **`scripts/vgpt2_v3/gap_analysis.py`**
   - Runs comprehensive tests on V3
   - Scores against required elements
   - Generates targeted training data
   - No API key required (offline)

2. **`scripts/vgpt2_v3/auto_improve.py`**
   - Full recursive improvement loop
   - Uses Claude/GPT-4 as "teacher" model
   - Generates harder questions each iteration
   - Tracks metrics and convergence
   - Requires ANTHROPIC_API_KEY or OPENAI_API_KEY

### Usage

```powershell
# Step 1: Run gap analysis (offline, no API needed)
python scripts/vgpt2_v3/gap_analysis.py

# Step 2: Review generated training data
cat data/vgpt2_v4_training.json

# Step 3: (Optional) Run automated loop with teacher model
$env:ANTHROPIC_API_KEY = "your-key"
python scripts/vgpt2_v3/auto_improve.py --iterations 5 --threshold 0.80

# Step 4: Train V4
llamafactory-cli train examples/train_lora/vgpt2_v4_sft.yaml
```

---

## V4 Training Configuration

```yaml
# examples/train_lora/vgpt2_v4_sft.yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
adapter_name_or_path: saves/vgpt2_v3/sft  # Continue from V3

# Training data - combine original + gap-filling
dataset: vgpt2_v3_sft,vgpt2_v4_training

# Same hyperparameters as V3
stage: sft
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: all

num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-4

output_dir: saves/vgpt2_v4/sft
```

---

## Realistic Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Run gap_analysis.py | 30 min |
| 2 | Review & expand training data | 2-4 hours |
| 3 | Train V4 SFT | 2-4 hours |
| 4 | Validate V4 | 30 min |
| 5 | (If needed) Generate more data | 1-2 hours |
| 6 | (If needed) Train V4.1 | 2-4 hours |
| **Total** | | **8-16 hours** |

---

## Success Criteria

V4 is "good enough" when:

1. ✅ Complex SQL: ≥75% (can produce aging, multi-table joins)
2. ✅ Business Logic: ≥75% (understands retainage, projections)
3. ✅ Hallucination: ≥90% (rejects fake tables reliably)
4. ✅ Cross-Module: ≥85% (knows join paths)
5. ✅ No regressions from V3 strengths

---

## Alternative: Teacher-Student Loop

If you have API access, the `auto_improve.py` script implements:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TEACHER-STUDENT LOOP                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STUDENT (VGPT2)              TEACHER (Claude/GPT-4)                │
│  ┌─────────────┐              ┌─────────────┐                       │
│  │ Answer Q    │──────────────│ Score A    │                       │
│  │             │              │ Provide    │                       │
│  │             │◄─────────────│ Reference  │                       │
│  └─────────────┘              └─────────────┘                       │
│         │                            │                              │
│         │      TRAINING DATA         │                              │
│         │     ┌─────────────┐        │                              │
│         └────►│ Q + Teacher │◄───────┘                              │
│               │ Answer      │                                       │
│               └──────┬──────┘                                       │
│                      │                                              │
│         ┌────────────▼────────────┐                                 │
│         │ FINE-TUNE STUDENT       │                                 │
│         │ ON GAP-FILLING DATA     │                                 │
│         └────────────┬────────────┘                                 │
│                      │                                              │
│                      ▼                                              │
│               REPEAT UNTIL CONVERGED                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

This is essentially **knowledge distillation** - extracting knowledge from a larger model into our smaller specialized one.

---

## Next Steps

1. **Run gap analysis now:**
   ```powershell
   python scripts/vgpt2_v3/gap_analysis.py
   ```

2. **Review the generated training data:**
   - Check `output/v3_gap_analysis.json` for detailed gaps
   - Check `data/vgpt2_v4_training.json` for training samples

3. **Expand training data if needed:**
   - Add variations of each sample
   - Include more edge cases

4. **Train V4 and validate**

Would you like me to run the gap analysis now?
