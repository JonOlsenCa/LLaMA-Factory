import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def probe(model_path: str, output_path: str) -> None:
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path not found: {model_dir}")

    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

    test_prompts = [
        "Generate SQL to list unpaid invoices with 30/60/90 aging buckets.\n\nDDL:\nCREATE TABLE ARTH (ARCo int, Mth datetime, ARTrans int, ARTransType char(1), CustGroup int, Customer int, TransDate datetime, PayFullDate datetime, Amount decimal(12,2));\nCREATE TABLE ARCM (CustGroup int, Customer int, Name varchar(60));\n",
        "Generate SQL to get subcontract cost breakdown with change orders and retainage.\n\nDDL:\nCREATE TABLE SLHD (SLCo int, SL int, Description varchar(80), Vendor int);\nCREATE TABLE SLIT (SLCo int, SL int, Item int, OrigCost decimal(12,2), CurCost decimal(12,2), InvCost decimal(12,2), PaidCost decimal(12,2), Retainage decimal(12,2));\n",
    ]

    results = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=256)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": text})
        print("---\n", text)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved probe output to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to SFT (or DPO) checkpoint directory")
    parser.add_argument("--output", required=True, help="Path to write probe results JSON")
    args = parser.parse_args()
    probe(args.model, args.output)


if __name__ == "__main__":
    main()
