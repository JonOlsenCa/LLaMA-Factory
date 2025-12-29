#!/usr/bin/env python3
"""
Quick test script for VGPT2 model
Usage: python scripts/test_vgpt2.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("=" * 60)
print("  VGPT2 Model Test")
print("=" * 60)

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "saves/vgpt2_v2_lora_sft"

print(f"\n[1/3] Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

print(f"[2/3] Loading LoRA adapter: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("[3/3] Running test queries...\n")
print("=" * 60)

# Test questions - comprehensive suite
test_questions = [
    # SQL Generation
    "Write a SQL query to get all invoices over $1000",
    "How do I join the Invoice and Vendor tables?",
    # Schema Knowledge
    "What tables are related to payments in AP Wizard?",
    "Describe the relationship between Document and DocumentDictionary",
    # Edge case - potentially unknown
    "What columns are in the UserPreferences table?",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n📝 Question {i}: {question}")
    print("-" * 60)
    
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"🤖 Response: {response.strip()}")
    print()

print("=" * 60)
print("✅ Test complete!")
print("=" * 60)

