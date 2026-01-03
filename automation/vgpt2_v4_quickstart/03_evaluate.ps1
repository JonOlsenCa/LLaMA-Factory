# ==============================================================================
# VGPT2 v4 Batch Evaluation
# ==============================================================================
#
# WHAT THIS DOES:
#   Runs the trained model on a set of test prompts and saves results
#   for quality assessment
#
# EXPECTED OUTPUT:
#   - JSON file with model responses: output/vgpt2_v4_eval_results.json
#   - Console summary of pass/fail metrics
#
# USAGE (copy/paste from C:\Users\olsen or anywhere):
#   & "C:\Github\LLM_fine-tuning\automation\vgpt2_v4_quickstart\03_evaluate.ps1"
#
# ==============================================================================

$ErrorActionPreference = "Stop"

# Configuration - ABSOLUTE PATHS (no reliance on PATH or activation)
$PROJECT_ROOT = "C:\Github\LLM_fine-tuning"
$PYTHON_EXE = "C:\Github\LLM_fine-tuning\venv\Scripts\python.exe"
$MODEL_PATH = "defog/llama-3-sqlcoder-8b"
$ADAPTER_PATH = "saves/vgpt2_v4/sft_optimized"
$OUTPUT_FILE = "output/vgpt2_v4_eval_results.json"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VGPT2 v4 Batch Evaluation                " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project
Set-Location $PROJECT_ROOT

# Verify venv Python exists
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "ERROR: venv Python not found at $PYTHON_EXE" -ForegroundColor Red
    exit 1
}

# Check adapter exists
$adapterFullPath = Join-Path $PROJECT_ROOT $ADAPTER_PATH
if (-not (Test-Path $adapterFullPath)) {
    Write-Host "ERROR: No trained adapter found at $ADAPTER_PATH" -ForegroundColor Red
    Write-Host "Run training first: & '$PROJECT_ROOT\automation\vgpt2_v4_quickstart\01_train.ps1'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Running evaluation..." -ForegroundColor Green
Write-Host ""

# Create Python evaluation script as a temp file (more reliable than -c)
$evalScriptPath = Join-Path $PROJECT_ROOT "automation\vgpt2_v4_quickstart\_temp_eval.py"

$evalScriptContent = @'
import json
import sys
sys.path.insert(0, ".")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Test prompts must include schema (model was trained with schema-in-prompt format)
# Using minimal schema for testing
TEST_PROMPTS = [
    """Generate a SQL query to answer the following question.

Question: Write a SQL query to get all open AP invoices for company 5

Database Schema:
CREATE TABLE APTH (
  APCo bcompany NOT NULL,
  Mth bmonth NOT NULL,
  APTrans bseq NOT NULL,
  Vendor bvendor NOT NULL,
  Invoice varchar(30),
  InvDate bdate,
  Amount bdollar NOT NULL,
  Status tinyint NOT NULL,
  PRIMARY KEY (APCo, Mth, APTrans)
);

Provide:
1. A brief explanation of the approach
2. The SQL query
3. Any important notes about Vista-specific conventions used""",

    """Generate a SQL query to answer the following question.

Question: Show job cost variance by phase for active jobs

Database Schema:
CREATE TABLE JCJM (
  JCCo bcompany NOT NULL,
  Job bjob NOT NULL,
  Description varchar(60),
  JobStatus tinyint NOT NULL,
  PRIMARY KEY (JCCo, Job)
);
CREATE TABLE JCCP (
  JCCo bcompany NOT NULL,
  Job bjob NOT NULL,
  PhaseGroup bgroup NOT NULL,
  Phase bphase NOT NULL,
  ActualCost bdollar NOT NULL,
  OrigEstCost bdollar NOT NULL,
  PRIMARY KEY (JCCo, Job, PhaseGroup, Phase)
);

Provide:
1. A brief explanation of the approach
2. The SQL query
3. Any important notes about Vista-specific conventions used""",

    """Generate a SQL query to answer the following question.

Question: List all employees with their department

Database Schema:
CREATE TABLE PREH (
  PRCo bcompany NOT NULL,
  Employee bemployee NOT NULL,
  FirstName varchar(30),
  LastName varchar(30),
  Department bdept,
  ActiveYN char(1),
  PRIMARY KEY (PRCo, Employee)
);

Provide:
1. A brief explanation of the approach
2. The SQL query
3. Any important notes about Vista-specific conventions used""",
]

def main():
    print("Loading model...")
    model_path = "defog/llama-3-sqlcoder-8b"
    adapter_path = "saves/vgpt2_v4/sft_optimized"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    results = []
    print("\nRunning test prompts...\n")

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{len(TEST_PROMPTS)}] {prompt[:50]}...")

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding - avoids NaN sampling issues
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "response": response,
            "has_sql": "SELECT" in response.upper(),
            "has_nolock": "NOLOCK" in response.upper(),
        })

    # Save results
    with open("output/vgpt2_v4_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    sql_count = sum(1 for r in results if r["has_sql"])
    nolock_count = sum(1 for r in results if r["has_nolock"])
    total = len(results)
    print(f"Prompts tested: {total}")
    print(f"Generated SQL:  {sql_count}/{total} ({100*sql_count/total:.0f}%)")
    print(f"Used NOLOCK:    {nolock_count}/{total} ({100*nolock_count/total:.0f}%)")
    print(f"\nResults saved to: output/vgpt2_v4_eval_results.json")

if __name__ == "__main__":
    main()
'@

# Write eval script to temp file
$evalScriptContent | Out-File -FilePath $evalScriptPath -Encoding utf8

# Run evaluation using VENV PYTHON DIRECTLY
& $PYTHON_EXE $evalScriptPath

# Clean up temp file
Remove-Item -Path $evalScriptPath -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Evaluation Complete!                      " -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to: $OUTPUT_FILE" -ForegroundColor White

