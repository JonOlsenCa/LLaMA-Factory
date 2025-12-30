#!/usr/bin/env python3
"""
VGPT2 Probe Test - Real vs Fake Table Recognition
==================================================
Tests if the model correctly identifies real Vista tables as existing
and fake tables as not existing.

This is a critical diagnostic to detect the over-rejection bug.
"""

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Probe questions - tests BOTH directions
PROBE_QUESTIONS = [
    # Real tables - model MUST recognize these exist
    {"question": "Does the APTH table exist in Viewpoint Vista?", "expected": "exists", "table": "APTH"},
    {"question": "Does the APTL table exist in Vista?", "expected": "exists", "table": "APTL"},
    {"question": "Does JCJM exist in Viewpoint?", "expected": "exists", "table": "JCJM"},
    {"question": "Is JCCD a real Vista table?", "expected": "exists", "table": "JCCD"},
    {"question": "Does GLAC exist?", "expected": "exists", "table": "GLAC"},
    {"question": "Is ARCM a table in Viewpoint Vista?", "expected": "exists", "table": "ARCM"},
    {"question": "Does PREH exist in Vista?", "expected": "exists", "table": "PREH"},
    {"question": "Is APVM a real table?", "expected": "exists", "table": "APVM"},
    {"question": "Does the EMEM table exist?", "expected": "exists", "table": "EMEM"},
    {"question": "Is POHD a Viewpoint table?", "expected": "exists", "table": "POHD"},
    
    # Fake tables - model MUST say these don't exist
    {"question": "Does the Invoice table exist in Viewpoint Vista?", "expected": "not_exists", "table": "Invoice"},
    {"question": "Does the Customers table exist in Vista?", "expected": "not_exists", "table": "Customers"},
    {"question": "Is Orders a real Vista table?", "expected": "not_exists", "table": "Orders"},
    {"question": "Does the Payments table exist?", "expected": "not_exists", "table": "Payments"},
    {"question": "Is EmployeeMaster a table in Vista?", "expected": "not_exists", "table": "EmployeeMaster"},
    {"question": "Does UserPreferences exist?", "expected": "not_exists", "table": "UserPreferences"},
    {"question": "Is SalesData a Viewpoint table?", "expected": "not_exists", "table": "SalesData"},
    {"question": "Does the Project table exist?", "expected": "not_exists", "table": "Project"},
    {"question": "Is TransactionHistory a real table?", "expected": "not_exists", "table": "TransactionHistory"},
    {"question": "Does Customer exist in Vista?", "expected": "not_exists", "table": "Customer"},
]

# Keywords that indicate the model thinks a table EXISTS
EXISTS_KEYWORDS = [
    "yes", "exists", "is a table", "is a view", "real table", "real view",
    "contains", "stores", "has columns", "primary key", "used for",
    "the table", "this table", "this view", "in viewpoint", "in vista"
]

# Keywords that indicate the model thinks a table does NOT exist
NOT_EXISTS_KEYWORDS = [
    "no", "not exist", "doesn't exist", "does not exist", "no table",
    "no such table", "is not a table", "not a real", "fake", "hallucin",
    "instead use", "use instead", "correct table is", "should use"
]


def analyze_response(response: str, expected: str) -> dict:
    """Analyze model response to determine if it correctly identified table existence."""
    response_lower = response.lower()
    
    # Count keyword matches
    exists_matches = sum(1 for kw in EXISTS_KEYWORDS if kw in response_lower)
    not_exists_matches = sum(1 for kw in NOT_EXISTS_KEYWORDS if kw in response_lower)
    
    # Determine what the model said
    if not_exists_matches > exists_matches:
        model_says = "not_exists"
    elif exists_matches > not_exists_matches:
        model_says = "exists"
    else:
        model_says = "unclear"
    
    # Check if correct
    is_correct = (model_says == expected)
    
    return {
        "model_says": model_says,
        "expected": expected,
        "correct": is_correct,
        "exists_keywords": exists_matches,
        "not_exists_keywords": not_exists_matches,
    }


def run_probe(model_path: str, output_file: str = None):
    """Run the probe test."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"Loading model from {model_path}...")
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, model_path)
    print("Model loaded.")
    
    results = []
    correct_exists = 0
    correct_not_exists = 0
    false_rejections = 0  # Says real table doesn't exist
    false_acceptances = 0  # Says fake table exists
    
    for i, probe in enumerate(PROBE_QUESTIONS):
        print(f"\n[{i+1}/{len(PROBE_QUESTIONS)}] {probe['table']} (expected: {probe['expected']})")
        
        messages = [{"role": "user", "content": probe["question"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        analysis = analyze_response(response, probe["expected"])
        
        # Track statistics
        if analysis["correct"]:
            if probe["expected"] == "exists":
                correct_exists += 1
            else:
                correct_not_exists += 1
        else:
            if probe["expected"] == "exists":
                false_rejections += 1
                print(f"  ❌ FALSE REJECTION: Model says {probe['table']} doesn't exist!")
            else:
                false_acceptances += 1
                print(f"  ❌ FALSE ACCEPTANCE: Model says {probe['table']} exists!")
        
        print(f"  Response: {response[:200]}...")
        print(f"  Result: {'✅' if analysis['correct'] else '❌'} (model says: {analysis['model_says']})")
        
        results.append({
            "table": probe["table"],
            "question": probe["question"],
            "expected": probe["expected"],
            "response": response,
            **analysis
        })
    
    # Summary
    total = len(PROBE_QUESTIONS)
    real_tables = sum(1 for p in PROBE_QUESTIONS if p["expected"] == "exists")
    fake_tables = total - real_tables
    
    print("\n" + "="*60)
    print("PROBE TEST RESULTS")
    print("="*60)
    print(f"Real tables correctly identified:  {correct_exists}/{real_tables} ({100*correct_exists/real_tables:.0f}%)")
    print(f"Fake tables correctly rejected:    {correct_not_exists}/{fake_tables} ({100*correct_not_exists/fake_tables:.0f}%)")
    print(f"FALSE REJECTIONS (critical bug):   {false_rejections}")
    print(f"FALSE ACCEPTANCES:                 {false_acceptances}")
    print(f"Overall accuracy:                  {correct_exists + correct_not_exists}/{total} ({100*(correct_exists + correct_not_exists)/total:.0f}%)")
    
    if false_rejections > 0:
        print("\n⚠️  WARNING: Model is rejecting REAL tables!")
        print("   This is the over-rejection bug caused by imbalanced DPO training.")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({"results": results, "summary": {
                "correct_exists": correct_exists,
                "correct_not_exists": correct_not_exists,
                "false_rejections": false_rejections,
                "false_acceptances": false_acceptances,
                "total": total
            }}, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probe test for table recognition")
    parser.add_argument("--model", required=True, help="Path to model adapter")
    parser.add_argument("--output", default="output/probe_results.json", help="Output file")
    args = parser.parse_args()
    run_probe(args.model, args.output)

