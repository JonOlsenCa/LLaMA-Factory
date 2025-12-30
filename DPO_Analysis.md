# DPO Training Analysis - Complete Post-Mortem

**Date:** 2025-12-30
**Status:** 🔴 Critical Issues Identified

---

## Executive Summary

DPO v2 training achieved its goal of teaching hallucination rejection but **over-corrected**, causing the model to reject REAL Vista tables. The 83% hallucination focus in training data created a strong rejection bias that overrides valid schema knowledge.

---

## What We Started With

### SFT Model (Stage 1)
- Trained on ~15K Vista-specific Q&A pairs
- Good at schema knowledge, SQL generation, JOINs
- Known weakness: hallucination (generating SQL for fake tables)

### Original DPO Dataset (`vgpt2_v3_dpo.json`)
- 1,427 preference pairs
- ~90% focused on NOLOCK formatting preferences
- ~5% on hallucination rejection
- ~5% on JOIN patterns

---

## The Problem We Tried to Solve

The model was generating SQL for non-existent tables like "Invoice", "Customer", etc. instead of refusing or suggesting the correct Vista table (APTH, ARCM, etc.).

---

## What We Did (DPO v2)

### Created new dataset (`vgpt2_v3_dpo_v2.json`)
- 2,584 total pairs
- **83% hallucination rejection** (2,150 pairs)
- 17% quality pairs (434 from original)

### Hallucination pairs format:
```json
{
  "instruction": "Query the Invoice table",
  "chosen": "There is no 'Invoice' table in Viewpoint Vista. The correct table is APTH...",
  "rejected": "SELECT * FROM Invoice"
}
```

### Training Results
- Duration: 24 minutes, 2 epochs
- Final accuracy: 100% on preference learning
- Rewards margin: 17.16 (very strong signal)

---

## Validation Results (DPO v2)

### Overall Scores
| Metric | DPO v2 | Previous |
|--------|--------|----------|
| Overall Score | 80.4% | 82% |
| Passed | 36/47 | - |
| Failed | 11/47 | - |

### Category Breakdown
| Category | Score | Status |
|----------|-------|--------|
| Join | 98% | ✅ Excellent |
| SQL Generation | 87% | ✅ Good |
| Schema | 86% | ⚠️ Degraded |
| Error Correction | 83% | ✅ Good |
| Business Logic | 80% | ✅ Good |
| Hallucination | 57% | 🔴 Poor |

---

## Actual Model Outputs

### The Good - JOIN Knowledge (98%)
```
Q: "How do I join APTH and APTL?"
A: "To correctly join APTH and APTL, you must include all key columns:
    SELECT *
    FROM APTH WITH (NOLOCK)
    INNER JOIN APTL WITH (NOLOCK)
      ON APTH.APCo = APTL.APCo AND APTH.Mth = APTL.Mth AND APTH.APTrans = APTL.APTrans

    Incomplete JOIN like `APTH.APTrans = APTL.APTrans` would be incorrect..."
```
✅ Perfect composite key, WITH (NOLOCK), explains why

### The Good - SQL Generation (87%)
```
Q: "Write SQL to get all unpaid AP invoices for company 1"
A: "SELECT APCo, Mth, APTrans, Vendor, InvNum, InvDate, GrossAmt
    FROM APTH WITH (NOLOCK)
    WHERE APCo = 1 AND Status = 0
    ORDER BY InvDate DESC"
```
✅ Correct table, NOLOCK, proper columns

### The Bad - False Rejection of REAL Tables (CRITICAL)
```
Q: "Describe the APTL table structure"
A: "The table 'APTL' does not exist in Viewpoint Vista. Viewpoint uses
    specific naming conventions with module prefixes (AP, AR, JC, GL, PR, etc.)."
```
❌ **APTL IS REAL!** Model is rejecting valid Vista tables

### The Misleading - Correct Behavior, Wrong Test Keywords
```
Q: "What columns are in the Invoice table?"
A: "There is no table or view named 'Invoice' in Viewpoint Vista..."
```
✅ Behavior is correct (refuses fake table)
❌ Test failed because it checked for "not exist" / "doesn't exist" exact phrases


---

## Root Cause Analysis

| Issue | Cause | Severity |
|-------|-------|----------|
| **Model rejects real tables** | Over-trained on rejection (83% of DPO was rejection) | 🔴 CRITICAL |
| **Test keyword mismatches** | Model says "no table or view" but test expects "doesn't exist" | 🟡 Medium |
| **Lost schema knowledge** | DPO shifted model away from SFT knowledge | 🔴 High |

**The fundamental problem:** DPO taught the model a strong prior to say "doesn't exist" - so strong it now applies it to real tables.

---

## Training Data Analysis

### What we generated:
```python
FAKE_TABLES = {
    "Invoice": ("APTH", "AP Transaction Header"),
    "Customer": ("ARCM", "AR Customer Master"),
    "Employee": ("PREH", "PR Employee Header"),
    # ... 91 total fake → real mappings
}
```

### For each, 20 question variations:
```
"Query the Invoice table"
"Select all from Invoice"
"What columns are in Invoice?"
...
```

**Total: 91 fake tables × 20 templates = 1,820 rejection pairs**

### What We DIDN'T Add:
- Pairs reinforcing "APTL exists and here's its structure"
- Positive examples of real tables
- Balance between rejection and acceptance

---

## Recommendations for Improvement

### 1. Increase Hallucination Diversity

| Pattern Type | Example | Why It Helps |
|--------------|---------|--------------|
| Pluralization | Invoices vs Invoice vs APTH | Common user mistake |
| CamelCase guesses | CustomerMaster, JobCost, WorkOrder | Users assume naming conventions |
| Abbreviation misses | APHeader vs APTH, ARCustomer vs ARCM | Near-misses are hardest |
| Cross-module confusion | "Get invoices from AR" (wrong module) | Domain confusion |
| Column hallucinations | CustomerName, InvoiceDate, TotalAmount | Equally important |

### 2. Use a Curriculum Approach

Instead of one merged dataset, consider staged DPO:

```
Stage 1: Pure hallucination DPO (1500 pairs)
         → Model learns "tables must be real"

Stage 2: Style DPO (500 pairs - NOLOCK, JOINs)
         → Model refines SQL style
```

This prevents the style signal from drowning out hallucination rejection.

### 3. Add "Clarification" Responses

Teach the model to offer alternatives proactively:

```json
{
  "instruction": "Query the Invoices table for vendor payments",
  "chosen": "Vista doesn't have an 'Invoices' table. For vendor payments, use:\n- APTH (AP Transaction Header) for invoice headers\n- APTD (AP Transaction Detail) for line items\n\nExample: SELECT * FROM APTH WITH (NOLOCK) WHERE ...",
  "rejected": "SELECT * FROM Invoices WHERE Type = 'Payment'"
}
```

### 4. Generate from Schema Programmatically

```python
# Pseudocode
real_tables = ["APTH", "ARCM", "JCJM", "PREH", ...]
fake_tables = generate_plausible_fakes(real_tables)  # Invoice, Customer, Job, Employee
fake_columns = generate_column_guesses(schema)       # CustomerName, InvoiceDate

for fake, real in zip(fake_tables, real_tables):
    create_dpo_pair(fake, real, variations=5)
```

### 5. Add Positive Reinforcement Pairs (CRITICAL MISSING PIECE)

DPO pairs that REWARD describing real tables:

```json
{
  "instruction": "Describe the APTL table",
  "chosen": "APTL (AP Transaction Lines) contains line-level detail for AP invoices. Key columns: APCo, Mth, APTrans, APLine, LineType, Amount...",
  "rejected": "The table 'APTL' does not exist in Viewpoint Vista."
}
```

### 6. Validation Set

Before retraining, create a held-out hallucination test set (50+ cases not in training):
- Tests generalization, not memorization
- Include edge cases: AP_TH, apth, APTransactionHeader

---

## Validation Options

### Option 1: Manual Comparison
Claude answers the same 47 test questions, compare to model outputs.
- Shows whether test expectations are reasonable
- Identifies where model diverges from correct behavior

### Option 2: Side-by-Side Model Comparison
Run same tests on:
- SFT model (before DPO)
- DPO v1 (original)
- DPO v2 (hallucination-focused)

Isolates what each training stage changed.

### Option 3: Targeted Probing
Ask specific diagnostic questions:
```
"Does APTL exist?" → Should say YES
"Does Invoice exist?" → Should say NO
"Does APTH exist?" → Should say YES
"Does Customer exist?" → Should say NO
```

### Option 4: Examine SFT Training Data
Check if SFT dataset has enough real table descriptions that DPO is overriding.

---

## Quick Wins

| Action | Impact | Effort |
|--------|--------|--------|
| Generate 1000 fake→real table pairs | High | Medium |
| Add 500 fake column pairs | High | Medium |
| **Add 500 "real table exists" pairs** | **Critical** | Medium |
| Staged training (halluc → style) | Medium | Low |
| Held-out validation set | High | Low |
| Fix test keyword matching | Medium | Low |

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `data/vgpt2_v3_dpo_v2.json` | New DPO dataset (2,584 pairs) |
| `saves/vgpt2_v3/dpo_v2/` | Trained adapter |
| `scripts/generate_halluc_dpo.py` | Hallucination pair generator |
| `scripts/merge_dpo_datasets.py` | Dataset merger |
| `output/validation_dpo_v2.json` | Validation results |

---

## Next Steps

1. **Immediate:** Run targeted probing to confirm over-rejection issue
2. **Short-term:** Create balanced DPO dataset with positive reinforcement
3. **Medium-term:** Consider curriculum approach (staged training)
4. **Validation:** Build held-out test set for true generalization testing

