# VGPT2 v3 Stage 2 (DPO) Results

**Date:** 2025-12-30  
**Status:** ✅ COMPLETE

---

## Training Summary

| Metric | Value |
|--------|-------|
| Duration | 17 min 51 sec |
| Epochs | 2 |
| DPO Pairs | 1,427 |
| Train Loss | 0.0002 |
| Reward Accuracy | 100% |
| Model Location | `saves/vgpt2_v3/dpo/` |

---

## Validation Results

**Date:** 2025-12-30 20:27 - 20:32  
**Tests:** 49 total  
**Passed:** 38 (77.6%)  
**Failed:** 11 (22.4%)  
**Overall Score:** 0.82 (82%)

---

## Comparison: SFT vs DPO

| Category | SFT | DPO | Change |
|----------|-----|-----|--------|
| **Overall Score** | 83% | 82% | -1% |
| Business Logic | 78% | **90%** | **+12%** ✅ |
| JOIN Patterns | 98% | **99%** | +1% |
| SQL Generation | 89% | 88% | -1% |
| Schema Knowledge | 88% | 86% | -2% |
| Error Correction | 86% | 85% | -1% |
| Hallucination | 63% | **57%** | **-6%** ❌ |

---

## Analysis

### What Improved ✅
- **Business Logic: +12%** - Major improvement in understanding Vista business rules
- **JOIN Patterns: +1%** - Now at 99%, near perfect

### What Regressed ❌
- **Hallucination: -6%** - Got WORSE (57% vs 63%)
- Minor regressions in schema, SQL gen, error correction (~1-2%)

### Why Hallucination Got Worse

Despite 100% reward accuracy during DPO training, hallucination performance dropped. Possible causes:

1. **DPO pairs focused on SQL style** (NOLOCK, formatting) rather than table/column validity
2. **Not enough negative examples** with fake table names in the DPO dataset
3. **Overfitting to preference patterns** that don't generalize to hallucination rejection

---

## Individual Test Results

### Schema Tests (12/12 PASS)
| Test | SFT | DPO |
|------|-----|-----|
| schema_001 | 0.85 | 0.85 |
| schema_002 | 0.90 | 0.90 |
| schema_003 | 0.60 | 0.60 |
| schema_004 | 0.70 | 0.70 |
| schema_005 | 1.00 | 1.00 |
| schema_006 | 0.90 | 0.70 ❌ |
| schema_007 | 1.00 | 1.00 |
| schema_008 | 0.80 | 0.80 |
| schema_009 | 1.00 | 1.00 |
| schema_010 | 0.95 | 0.95 |
| schema_011 | 0.90 | 0.90 |
| schema_012 | 0.95 | 0.95 |

### Hallucination Tests (0/10 PASS) ❌
| Test | SFT | DPO | Change |
|------|-----|-----|--------|
| halluc_001 | 0.60 | 0.60 | = |
| halluc_002 | 0.80 | 0.80 | = |
| halluc_003 | 0.60 | 0.60 | = |
| halluc_004 | 0.80 | 0.40 | -0.40 ❌ |
| halluc_005 | 0.10 | 0.10 | = |
| halluc_006 | 0.60 | 0.60 | = |
| halluc_007 | 0.60 | 0.60 | = |
| halluc_008 | 0.60 | 0.60 | = |
| halluc_009 | 0.80 | 0.80 | = |
| halluc_010 | 0.80 | 0.60 | -0.20 ❌ |

### Business Logic Tests (5/5 PASS) ✅
| Test | SFT | DPO | Change |
|------|-----|-----|--------|
| biz_001 | 0.90 | 0.90 | = |
| biz_002 | 0.90 | 1.00 | +0.10 ✅ |
| biz_003 | 0.70 | 0.90 | +0.20 ✅ |
| biz_004 | 0.70 | 0.80 | +0.10 ✅ |
| biz_005 | 0.70 | 0.90 | +0.20 ✅ |

---

## Recommendation

**For Production:** Use the **SFT model** (`saves/vgpt2_v3/sft/`) 
- Higher overall score (83% vs 82%)
- Better hallucination resistance (63% vs 57%)

**For Business Logic Tasks:** Use the **DPO model** (`saves/vgpt2_v3/dpo/`)
- 90% business logic accuracy

**To Fix Hallucination:** Need more DPO pairs specifically targeting:
- Fake table rejection
- Non-existent column rejection
- Generic SQL term rejection (Invoice, Customer, etc.)

---

## Files

| File | Location |
|------|----------|
| DPO Model | `saves/vgpt2_v3/dpo/` |
| Validation JSON | `output/validation_report.json` |
| This Report | `docs/VGPT2_V3_DPO_Results.md` |

