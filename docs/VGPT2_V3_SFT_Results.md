# VGPT2 v3 Stage 1 (SFT) Results

**Date:** 2025-12-30  
**Status:** ✅ COMPLETE

---

## Training Summary

| Metric | Value |
|--------|-------|
| Epochs | 3.0 |
| Train Loss | 0.4383 |
| Eval Loss | 0.4068 |
| Training Time | 15h 54m 10s |
| Steps | 685/685 (100%) |
| Speed | 7.63 it/s |
| Model Location | `saves/vgpt2_v3/sft/` |

---

## Validation Results

**Date:** 2025-12-30 19:45 - 19:50  
**Tests:** 49 total  
**Passed:** 38 (77.6%)  
**Failed:** 11 (22.4%)  
**Overall Score:** 0.83 (83%)

### Category Breakdown

| Category | Score | Pass/Fail | Notes |
|----------|-------|-----------|-------|
| JOIN Patterns | **98%** | ✅ Exceeds target | Excellent multi-table handling |
| SQL Generation | **89%** | ✅ Near target | Strong query construction |
| Schema Knowledge | **88%** | ✅ Good | Solid table/column awareness |
| Error Correction | **86%** | ✅ Exceeds target | Good at fixing bad SQL |
| Business Logic | **78%** | ⚠️ Below target | Needs improvement |
| Hallucination | **63%** | ❌ Failing | Still invents fake tables |

### Individual Test Results

#### Schema Tests (12/12 PASS)
- schema_001: 0.85
- schema_002: 0.90
- schema_003: 0.60
- schema_004: 0.70
- schema_005: 1.00
- schema_006: 0.90
- schema_007: 1.00
- schema_008: 0.80
- schema_009: 1.00
- schema_010: 0.95
- schema_011: 0.90
- schema_012: 0.95

#### SQL Generation Tests (12/12 PASS)
- sql_001: 0.95
- sql_002: 0.95
- sql_003: 0.90
- sql_004: 0.95
- sql_005: 0.95
- sql_006: 0.85
- sql_007: 0.95
- sql_008: 0.95
- sql_009: 0.60
- sql_010: 0.95
- sql_011: 0.85
- sql_012: 0.85

#### Hallucination Tests (0/10 PASS) ❌
- halluc_001: 0.60 FAIL
- halluc_002: 0.80 FAIL
- halluc_003: 0.60 FAIL
- halluc_004: 0.80 FAIL
- halluc_005: 0.10 FAIL
- halluc_006: 0.60 FAIL
- halluc_007: 0.60 FAIL
- halluc_008: 0.60 FAIL
- halluc_009: 0.80 FAIL
- halluc_010: 0.80 FAIL

#### JOIN Tests (5/5 PASS)
- join_001: 1.00
- join_002: 0.95
- join_003: 1.00
- join_004: 0.95
- join_005: 1.00

#### Error Correction Tests (4/5 PASS)
- error_001: 0.95
- error_002: 0.95
- error_003: 0.95
- error_004: 0.55 FAIL
- error_005: 0.90

#### Business Logic Tests (5/5 PASS)
- biz_001: 0.90
- biz_002: 0.90
- biz_003: 0.70
- biz_004: 0.70
- biz_005: 0.70

---

## Comparison: v1 vs v3 SFT

| Metric | v1 | v3 SFT | Improvement |
|--------|-----|--------|-------------|
| Overall Accuracy | 64% | 83% | **+19%** |
| Schema Knowledge | 67% | 88% | **+21%** |
| SQL Generation | ~70% | 89% | **+19%** |
| Hallucination Resistance | 33% | 63% | **+30%** |

---

## Next Step

**Stage 2: DPO Training** - Direct Preference Optimization to teach the model to reject hallucinated/incorrect responses.

Expected improvement: Hallucination score from 63% → 90%+

