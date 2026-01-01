#!/usr/bin/env python3
"""
VGPT2 Gap Analysis and V4 Training Data Generator

This script:
1. Runs comprehensive gap analysis across all categories
2. Identifies specific weaknesses with examples
3. Generates targeted training data for V4

Usage: python scripts/vgpt2_v3/gap_analysis.py
"""

import json
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Known gaps from testing + comprehensive new tests
COMPREHENSIVE_TESTS = {
    # ========== COMPLEX SQL (Current score: 35.6%) ==========
    "complex_sql": {
        "description": "Multi-table joins, CTEs, CASE WHEN, aggregations",
        "tests": [
            {
                "id": "cs001",
                "question": "Write SQL to calculate AR aging buckets (30/60/90+ days) for unpaid invoices by customer",
                "required_elements": ["ARTH", "ARCM", "DATEDIFF", "CASE WHEN", "PayFullDate IS NULL", "CustGroup", "30", "60", "90"],
                "ground_truth": """```sql
SELECT 
    ARTH.ARCo, ARCM.CustGroup, ARCM.Customer, ARCM.Name,
    SUM(CASE WHEN DATEDIFF(day, ARTH.InvDate, GETDATE()) <= 30 THEN ARTH.Amount ELSE 0 END) AS [0-30 Days],
    SUM(CASE WHEN DATEDIFF(day, ARTH.InvDate, GETDATE()) BETWEEN 31 AND 60 THEN ARTH.Amount ELSE 0 END) AS [31-60 Days],
    SUM(CASE WHEN DATEDIFF(day, ARTH.InvDate, GETDATE()) BETWEEN 61 AND 90 THEN ARTH.Amount ELSE 0 END) AS [61-90 Days],
    SUM(CASE WHEN DATEDIFF(day, ARTH.InvDate, GETDATE()) > 90 THEN ARTH.Amount ELSE 0 END) AS [Over 90 Days]
FROM ARTH WITH (NOLOCK)
INNER JOIN ARCM WITH (NOLOCK) 
    ON ARTH.CustGroup = ARCM.CustGroup AND ARTH.Customer = ARCM.Customer
WHERE ARTH.ARCo = @ARCo AND ARTH.PayFullDate IS NULL
GROUP BY ARTH.ARCo, ARCM.CustGroup, ARCM.Customer, ARCM.Name
ORDER BY ARCM.Customer
```""",
            },
            {
                "id": "cs002",
                "question": "Write SQL to aggregate job cost estimates by phase and cost type with item vs phase unit distinctions",
                "required_elements": ["JCJP", "JCCH", "JCCP", "ItemUnitFlag", "PhaseUnitFlag", "OrigEstCost", "CurrEstCost"],
                "ground_truth": """```sql
SELECT 
    JCCP.JCCo, JCCP.Job, JCCP.PhaseGroup, JCCP.Phase, JCCP.CostType,
    JCCH.ItemUnitFlag, JCCH.PhaseUnitFlag,
    SUM(JCCP.OrigEstCost) AS OrigEstCost,
    SUM(JCCP.CurrEstCost) AS CurrEstCost,
    SUM(JCCP.CurrEstHours) AS CurrEstHours,
    CASE WHEN JCCH.ItemUnitFlag = 'Y' THEN 'Item Units' ELSE 'Phase Units' END AS UnitType
FROM JCCP WITH (NOLOCK)
INNER JOIN JCCH WITH (NOLOCK) 
    ON JCCP.JCCo = JCCH.JCCo AND JCCP.CostType = JCCH.CostType
INNER JOIN JCJP WITH (NOLOCK)
    ON JCCP.JCCo = JCJP.JCCo AND JCCP.Job = JCJP.Job 
    AND JCCP.PhaseGroup = JCJP.PhaseGroup AND JCCP.Phase = JCJP.Phase
WHERE JCCP.JCCo = @JCCo AND JCCP.Job = @Job
GROUP BY JCCP.JCCo, JCCP.Job, JCCP.PhaseGroup, JCCP.Phase, JCCP.CostType,
    JCCH.ItemUnitFlag, JCCH.PhaseUnitFlag
```""",
            },
            {
                "id": "cs003",
                "question": "Write SQL to track AP hold status distinguishing retainage vs non-retainage holds",
                "required_elements": ["APTD", "APHD", "APCO", "RetHoldCode", "HoldCode", "PayType"],
                "ground_truth": """```sql
SELECT 
    APTD.APCo, APTD.Mth, APTD.APTrans, APTD.APLine, APTD.PayType,
    APHD.HoldCode, APCO.RetHoldCode,
    CASE WHEN APHD.HoldCode = APCO.RetHoldCode THEN 'Retainage Hold' 
         ELSE 'Non-Retainage Hold' END AS HoldType,
    APTD.Amount
FROM APTD WITH (NOLOCK)
INNER JOIN APHD WITH (NOLOCK)
    ON APTD.APCo = APHD.APCo AND APTD.Mth = APHD.Mth 
    AND APTD.APTrans = APHD.APTrans AND APTD.APLine = APHD.APLine
INNER JOIN APCO WITH (NOLOCK)
    ON APTD.APCo = APCO.APCo
WHERE APTD.APCo = @APCo AND APHD.HoldCode IS NOT NULL
```""",
            },
            {
                "id": "cs004",
                "question": "Write SQL to reconcile AP oncost batch lines with original transactions",
                "required_elements": ["APLB", "APHB", "APTL", "APTH", "ocApplyMth", "ocApplyTrans", "ocApplyLine"],
                "ground_truth": """```sql
SELECT 
    APLB.Co, APLB.Mth, APLB.BatchId, APLB.BatchSeq, APLB.APLine,
    APLB.ocApplyMth, APLB.ocApplyTrans, APLB.ocApplyLine,
    APTL_Orig.Description AS OriginalDescription,
    APTL_Orig.GrossAmt AS OriginalAmount,
    APLB.GrossAmt AS OnCostAmount
FROM APLB WITH (NOLOCK)
INNER JOIN APHB WITH (NOLOCK)
    ON APLB.Co = APHB.Co AND APLB.Mth = APHB.Mth 
    AND APLB.BatchId = APHB.BatchId AND APLB.BatchSeq = APHB.BatchSeq
INNER JOIN APTL APTL_Orig WITH (NOLOCK)
    ON APLB.Co = APTL_Orig.APCo 
    AND APLB.ocApplyMth = APTL_Orig.Mth 
    AND APLB.ocApplyTrans = APTL_Orig.APTrans 
    AND APLB.ocApplyLine = APTL_Orig.APLine
WHERE APLB.Co = @APCo AND APLB.ocApplyMth IS NOT NULL
```""",
            },
            {
                "id": "cs005",
                "question": "Write SQL with CTE to calculate running total of job costs by month",
                "required_elements": ["CTE", "WITH", "SUM", "OVER", "PARTITION BY", "ORDER BY", "JCCD"],
                "ground_truth": """```sql
WITH MonthlyCosts AS (
    SELECT 
        JCCo, Job, Mth,
        SUM(ActualCost) AS MonthCost
    FROM JCCD WITH (NOLOCK)
    WHERE JCCo = @JCCo AND Job = @Job
    GROUP BY JCCo, Job, Mth
)
SELECT 
    JCCo, Job, Mth, MonthCost,
    SUM(MonthCost) OVER (PARTITION BY JCCo, Job ORDER BY Mth) AS RunningTotal
FROM MonthlyCosts
ORDER BY Mth
```""",
            },
        ],
    },
    
    # ========== BUSINESS LOGIC (Current score: 34.1%) ==========
    "business_logic": {
        "description": "Viewpoint-specific calculations, workflows, field meanings",
        "tests": [
            {
                "id": "bl001",
                "question": "How does Vista calculate maximum retainage when InclACOinMaxYN is set to Y?",
                "required_elements": ["InclACOinMaxYN", "MaxRetgPct", "CurCost", "OrigCost", "SLHB", "SLCO"],
                "ground_truth": """When InclACOinMaxYN='Y', Vista includes Approved Change Orders in the maximum retainage calculation:

MaxRetainage = MaxRetgPct * CurCost

When InclACOinMaxYN='N', it uses only original costs:
MaxRetainage = MaxRetgPct * OrigCost

This setting is found in SLHB (Subcontract Header Billing) and SLCO (Subcontract Company) tables. The CurCost includes OrigCost plus all approved change order amounts.""",
            },
            {
                "id": "bl002",
                "question": "What is the difference between WCRetAmt and SMRetAmt in subcontract worksheets?",
                "required_elements": ["WCRetAmt", "SMRetAmt", "SLWI", "Work Completed", "Stored Materials"],
                "ground_truth": """In the SLWI (Subcontract Worksheet Item) table:

- **WCRetAmt** = Work Completed Retainage - Retainage withheld on labor and work in place
- **SMRetAmt** = Stored Materials Retainage - Retainage withheld on materials stored on-site but not yet installed

Total Retainage = WCRetAmt + SMRetAmt

Different retainage percentages may apply:
- WCRetPct = Work Completed Retainage Percentage
- SMRetPct = Stored Materials Retainage Percentage

This allows contractors to apply different retainage rates to work vs materials.""",
            },
            {
                "id": "bl003",
                "question": "What tables are involved in cost-to-complete projections for jobs?",
                "required_elements": ["JCPR", "JCPD", "JCCP", "JCCD", "CurrEstCost", "ActualCost"],
                "ground_truth": """Key tables for cost-to-complete projections:

1. **JCPR** (JC Projection Resources) - Stores projection settings and resources
2. **JCPD** (JC Projection Detail) - Stores projection detail amounts by phase/cost type
3. **JCCP** (JC Cost Phase) - Contains current estimates (CurrEstCost, CurrEstHours)
4. **JCCD** (JC Cost Detail) - Contains actual costs (ActualCost, ActualHours)

Projection calculation:
- Cost to Complete = CurrEstCost - ActualCost
- Percent Complete = ActualCost / CurrEstCost

The JCPR/JCPD tables store user-adjusted projections that may differ from the standard calculation.""",
            },
            {
                "id": "bl004",
                "question": "How does the duplicate invoice detection work in AP?",
                "required_elements": ["APTH", "Vendor", "APRef", "InvDate", "udDuplicateInvoice", "APCO"],
                "ground_truth": """Viewpoint's duplicate invoice detection checks multiple criteria:

1. **Company + Vendor + APRef match** - Same vendor, same invoice reference
2. **Date proximity** - Invoices within a configurable date range
3. **Amount comparison** - Matching gross amounts

The check uses APTH table with these key columns:
- APCo (company)
- VendorGroup, Vendor (vendor identification)  
- APRef (invoice reference/number)
- InvDate (invoice date)
- GrossAmt (invoice amount)

APCO company settings control duplicate checking behavior:
- DupInvChk (Y/N to enable)
- DupInvDays (date range to check)

Custom field udDuplicateInvoice may flag duplicates for workflow routing.""",
            },
            {
                "id": "bl005",
                "question": "Explain how stored materials (Purchased - Installed) affects SL billing",
                "required_elements": ["SLWI", "StoredMatls", "WCCost", "SMRetPct", "WCRetPct", "Purchased", "Installed"],
                "ground_truth": """In SLWI (Subcontract Worksheet Item):

**StoredMatls = Purchased - Installed**

This represents materials on-site but not yet used in the work.

Billing impact:
- **WCCost** = Work Completed Cost (labor and installed materials)
- **StoredMatls** = Materials purchased but not installed
- **Total Billable** = WCCost + StoredMatls

Different retainage rates apply:
- **WCRetPct** = Retainage percentage for work completed
- **SMRetPct** = Retainage percentage for stored materials

The subcontractor can bill for materials stored on-site even before installation, but typically at a different retainage rate than completed work.""",
            },
        ],
    },
    
    # ========== CROSS-MODULE JOINS (Current score: 72.6%) ==========
    "cross_module_joins": {
        "description": "Complex join paths across AP, AR, JC, GL, SL, PR modules",
        "tests": [
            {
                "id": "xm001",
                "question": "How do I join SLWI retainage amounts to matching APTD transactions?",
                "required_elements": ["SLWI", "APTL", "APTD", "SLCo", "SL", "SLItem", "APLine", "RetPayType"],
                "ground_truth": """Join path: SLWI → APTL → APTD

```sql
SELECT 
    SLWI.SLCo, SLWI.SL, SLWI.SLItem, 
    SLWI.WCRetAmt, SLWI.SMRetAmt,
    APTD.Amount AS APRetainageAmt
FROM SLWI WITH (NOLOCK)
INNER JOIN APTL WITH (NOLOCK)
    ON SLWI.SLCo = APTL.APCo 
    AND SLWI.SL = APTL.SL 
    AND SLWI.SLItem = APTL.SLItem
INNER JOIN APTD WITH (NOLOCK)
    ON APTL.APCo = APTD.APCo 
    AND APTL.Mth = APTD.Mth 
    AND APTL.APTrans = APTD.APTrans 
    AND APTL.APLine = APTD.APLine
INNER JOIN APCO WITH (NOLOCK)
    ON APTD.APCo = APCO.APCo
WHERE APTD.PayType = APCO.RetPayType  -- Filter for retainage payments only
```""",
            },
            {
                "id": "xm002",
                "question": "How do I trace a GL journal entry back to its source AP invoice?",
                "required_elements": ["GLDT", "APTH", "APTL", "Source", "SourceCo", "Mth", "APTrans"],
                "ground_truth": """Join path: GLDT → APTH (via Source columns)

```sql
SELECT 
    GLDT.GLCo, GLDT.Mth, GLDT.GLTrans, GLDT.GLAcct,
    GLDT.Source, GLDT.SourceCo,
    APTH.Vendor, APTH.APRef, APTH.Description,
    GLDT.Amount
FROM GLDT WITH (NOLOCK)
INNER JOIN APTH WITH (NOLOCK)
    ON GLDT.SourceCo = APTH.APCo 
    AND GLDT.Mth = APTH.Mth 
    AND GLDT.Source = 'AP'
    AND GLDT.BatchId = APTH.BatchId
    AND GLDT.BatchSeq = APTH.BatchSeq
WHERE GLDT.GLCo = @GLCo AND GLDT.Source = 'AP'
```

Note: Source='AP' identifies AP-originated entries. The BatchId/BatchSeq or APTrans link varies by posting method.""",
            },
            {
                "id": "xm003",
                "question": "Build an audit trail from PR timecard to JC cost to GL posting",
                "required_elements": ["PRTH", "JCCD", "GLDT", "PRCo", "Employee", "Job", "Phase", "Source"],
                "ground_truth": """Full audit trail: PRTH → JCCD → GLDT

```sql
SELECT 
    PRTH.PRCo, PRTH.Employee, PRTH.PREndDate,
    PRTH.Job, PRTH.Phase, PRTH.Hours,
    JCCD.ActualCost, JCCD.ActualHours,
    GLDT.GLAcct, GLDT.Amount AS GLAmount
FROM PRTH WITH (NOLOCK)
INNER JOIN JCCD WITH (NOLOCK)
    ON PRTH.JCCo = JCCD.JCCo 
    AND PRTH.Job = JCCD.Job 
    AND PRTH.PhaseGroup = JCCD.PhaseGroup
    AND PRTH.Phase = JCCD.Phase
    AND PRTH.PRCo = JCCD.PRCo 
    AND PRTH.Employee = JCCD.Employee
    AND PRTH.PREndDate = JCCD.ActualDate
INNER JOIN GLDT WITH (NOLOCK)
    ON JCCD.JCCo = GLDT.SourceCo 
    AND JCCD.Mth = GLDT.Mth
    AND GLDT.Source = 'JC'
WHERE PRTH.PRCo = @PRCo AND PRTH.Employee = @Employee
```""",
            },
        ],
    },
    
    # ========== HALLUCINATION DETECTION (Current score: 10%) ==========
    "hallucination": {
        "description": "Correctly reject non-existent tables, suggest alternatives",
        "tests": [
            {
                "id": "hd001",
                "question": "What columns are in the ARAgingReport table?",
                "required_elements": ["does not exist", "ARTH", "ARTD", "vrvAR"],
                "ground_truth": """The table 'ARAgingReport' does not exist in Viewpoint Vista.

For AR aging data, use these actual tables/views:
- **ARTH** (AR Transaction Header) - Invoice header with InvDate, Amount
- **ARTD** (AR Transaction Detail) - Line-level details
- **vrvAR_MVAllInvoices** - Reporting view with invoice details

Example aging query:
```sql
SELECT ARCo, Customer, InvDate, Amount,
    DATEDIFF(day, InvDate, GETDATE()) AS DaysOutstanding
FROM ARTH WITH (NOLOCK)
WHERE PayFullDate IS NULL
```""",
            },
            {
                "id": "hd002",
                "question": "Write SQL to query the SubcontractorPayments table",
                "required_elements": ["does not exist", "APTD", "APTL", "SLWI", "SL"],
                "ground_truth": """The table 'SubcontractorPayments' does not exist in Viewpoint Vista.

For subcontractor payment data, use these actual tables:
- **APTD** (AP Transaction Detail) - Payment details linked to subcontracts
- **APTL** (AP Transaction Line) - Line items with SL reference
- **SLWI** (Subcontract Worksheet Item) - Worksheet payment tracking

Example query:
```sql
SELECT APTL.APCo, APTL.SL, APTL.SLItem, APTD.Amount, APTD.PaidDate
FROM APTL WITH (NOLOCK)
INNER JOIN APTD WITH (NOLOCK)
    ON APTL.APCo = APTD.APCo AND APTL.Mth = APTD.Mth 
    AND APTL.APTrans = APTD.APTrans AND APTL.APLine = APTD.APLine
WHERE APTL.SL IS NOT NULL AND APTD.Status = 4
```""",
            },
            {
                "id": "hd003",
                "question": "How do I join the VendorList and InvoiceHeader tables?",
                "required_elements": ["does not exist", "do not exist", "APVM", "APTH", "VendorGroup", "Vendor"],
                "ground_truth": """Neither 'VendorList' nor 'InvoiceHeader' exist in Viewpoint Vista.

The correct Viewpoint tables are:
- **APVM** (AP Vendor Master) - Vendor information
- **APTH** (AP Transaction Header) - Invoice headers

Correct join:
```sql
SELECT APVM.Vendor, APVM.Name, APTH.APRef, APTH.InvDate, APTH.GrossAmt
FROM APTH WITH (NOLOCK)
INNER JOIN APVM WITH (NOLOCK)
    ON APTH.VendorGroup = APVM.VendorGroup AND APTH.Vendor = APVM.Vendor
WHERE APTH.APCo = @APCo
```

Viewpoint uses module prefixes (AP, AR, JC, etc.) not generic names.""",
            },
            {
                "id": "hd004",
                "question": "What is the PaymentStatus column in the APTH table?",
                "required_elements": ["does not exist", "Status", "PayFullDate", "PaidMth"],
                "ground_truth": """There is no 'PaymentStatus' column in APTH.

The correct columns for payment status in APTH are:
- **Status** - Numeric status (0=Open, 1=Hold, 2=Partial, 3=In Progress, 4=Paid)
- **PayFullDate** - Date invoice was fully paid (NULL if unpaid)
- **PaidMth** - Month when payment was posted

Example:
```sql
SELECT APCo, APRef, 
    CASE Status 
        WHEN 0 THEN 'Open'
        WHEN 4 THEN 'Paid'
        ELSE 'In Process'
    END AS PaymentStatus
FROM APTH WITH (NOLOCK)
WHERE APCo = @APCo
```""",
            },
        ],
    },
}


def load_model(adapter_path: str = "saves/vgpt2_v3/sft"):
    """Load VGPT2 model."""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Model loaded!")
    return model, tokenizer


def query_model(model, tokenizer, question: str) -> str:
    """Get model response."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    ).strip()


def score_response(response: str, required_elements: list) -> dict:
    """Score response against required elements."""
    response_lower = response.lower()
    
    found = []
    missing = []
    
    for elem in required_elements:
        if elem.lower() in response_lower:
            found.append(elem)
        else:
            missing.append(elem)
    
    score = len(found) / len(required_elements) if required_elements else 0
    
    return {
        "score": score,
        "found": found,
        "missing": missing,
    }


def run_gap_analysis():
    """Run comprehensive gap analysis."""
    model, tokenizer = load_model()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "categories": {},
        "all_gaps": [],
        "training_data": [],
    }
    
    for category, data in COMPREHENSIVE_TESTS.items():
        print(f"\n{'='*70}")
        print(f"📁 Category: {category.upper()}")
        print(f"   {data['description']}")
        print("="*70)
        
        category_scores = []
        category_gaps = []
        
        for test in data["tests"]:
            print(f"\n[{test['id']}] {test['question'][:60]}...")
            
            # Get model response
            response = query_model(model, tokenizer, test["question"])
            print(f"Response: {response[:100]}...")
            
            # Score it
            scoring = score_response(response, test["required_elements"])
            print(f"Score: {scoring['score']:.0%}")
            print(f"Found: {scoring['found']}")
            print(f"Missing: {scoring['missing']}")
            
            category_scores.append(scoring["score"])
            
            # If score < 0.7, it's a gap - generate training data
            if scoring["score"] < 0.7:
                gap = {
                    "id": test["id"],
                    "category": category,
                    "question": test["question"],
                    "score": scoring["score"],
                    "missing_elements": scoring["missing"],
                    "model_response": response,
                }
                category_gaps.append(gap)
                results["all_gaps"].append(gap)
                
                # Generate training sample
                training_sample = {
                    "instruction": test["question"],
                    "input": "",
                    "output": test["ground_truth"],
                    "category": category,
                    "gap_id": test["id"],
                }
                results["training_data"].append(training_sample)
        
        # Category summary
        avg_score = sum(category_scores) / len(category_scores) if category_scores else 0
        results["categories"][category] = {
            "average_score": avg_score,
            "tests_run": len(data["tests"]),
            "gaps_found": len(category_gaps),
        }
        
        print(f"\n📊 {category} Summary: {avg_score:.0%} ({len(category_gaps)} gaps)")
    
    # Overall summary
    all_scores = [r["categories"][c]["average_score"] for c in results["categories"]]
    results["overall_score"] = sum(all_scores) / len(all_scores)
    
    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save gap analysis
    gaps_path = output_dir / "v3_gap_analysis.json"
    with open(gaps_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved gap analysis to {gaps_path}")
    
    # Save training data for V4
    if results["training_data"]:
        training_path = Path("data") / "vgpt2_v4_training.json"
        with open(training_path, 'w', encoding='utf-8') as f:
            json.dump(results["training_data"], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results['training_data'])} training samples to {training_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("  GAP ANALYSIS SUMMARY")
    print("="*70)
    print(f"  Overall Score: {results['overall_score']:.0%}")
    print(f"  Total Gaps: {len(results['all_gaps'])}")
    print(f"  Training Samples Generated: {len(results['training_data'])}")
    print("\n  Category Breakdown:")
    for cat, data in results["categories"].items():
        status = "✅" if data["average_score"] >= 0.7 else "❌"
        print(f"    {status} {cat}: {data['average_score']:.0%} ({data['gaps_found']} gaps)")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_gap_analysis()
