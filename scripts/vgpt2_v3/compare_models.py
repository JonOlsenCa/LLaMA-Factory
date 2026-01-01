#!/usr/bin/env python3
"""
Compare Model Responses Against Ground Truth
=============================================
This script tests the newly trained DPO v2 model against the same questions
that were answered by Opus (ground truth), and provides a detailed comparison.

Usage:
    python scripts/vgpt2_v3/compare_models.py --model saves/vgpt2_v3/dpo_v2
    python scripts/vgpt2_v3/compare_models.py --model saves/vgpt2_v3/dpo_v2 --compare-sft
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load model with LoRA adapter."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate a response for a question."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def parse_ground_truth_md(md_path: str) -> List[Dict]:
    """Parse the GROUND_TRUTH_ANSWERS.md file into structured questions."""
    questions = []
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by ### headers (each question)
    sections = re.split(r'\n###\s+', content)
    
    for section in sections[1:]:  # Skip intro
        lines = section.strip().split('\n')
        if not lines:
            continue
        
        # Parse question ID and text
        first_line = lines[0]
        match = re.match(r'(\w+):\s*(.+)', first_line)
        if not match:
            continue
        
        q_id = match.group(1)
        question = match.group(2)
        
        # Find expected keywords
        expected_keywords = []
        forbidden_keywords = []
        ground_truth = ""
        
        in_ground_truth = False
        gt_lines = []
        
        for line in lines[1:]:
            if line.startswith('**Expected Keywords:**'):
                keywords_str = line.replace('**Expected Keywords:**', '').strip()
                expected_keywords = [k.strip() for k in keywords_str.split(',')]
            elif line.startswith('**Forbidden Keywords:**'):
                keywords_str = line.replace('**Forbidden Keywords:**', '').strip()
                forbidden_keywords = [k.strip() for k in keywords_str.split(',')]
            elif line.startswith('**Ground Truth:**'):
                in_ground_truth = True
            elif in_ground_truth:
                if line.startswith('---') or line.startswith('### '):
                    break
                gt_lines.append(line)
        
        ground_truth = '\n'.join(gt_lines).strip()
        
        if question and ground_truth:
            questions.append({
                "id": q_id,
                "question": question,
                "expected_keywords": expected_keywords,
                "forbidden_keywords": forbidden_keywords,
                "ground_truth": ground_truth
            })
    
    return questions


def load_comprehensive_test_suite(json_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load test questions from the comprehensive JSON suite."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        questions.append({
            "id": item.get("id", "unknown"),
            "category": item.get("category", "general"),
            "question": item.get("question", ""),
            "ground_truth": item.get("ground_truth", ""),
            "key_elements": item.get("key_elements", [])
        })
    
    if limit:
        questions = questions[:limit]
    
    return questions


def score_response(response: str, ground_truth: str, key_elements: List[str], 
                   forbidden_keywords: List[str] = None) -> Dict:
    """Score a response against ground truth and key elements."""
    response_lower = response.lower()
    ground_truth_lower = ground_truth.lower()
    
    # Score key elements
    found = []
    missing = []
    for element in key_elements:
        if element.lower() in response_lower:
            found.append(element)
        else:
            missing.append(element)
    
    element_score = len(found) / len(key_elements) if key_elements else 1.0
    
    # Check forbidden keywords
    forbidden_found = []
    if forbidden_keywords:
        for kw in forbidden_keywords:
            if kw.lower() in response_lower:
                forbidden_found.append(kw)
    
    # Penalize for forbidden keywords
    forbidden_penalty = 0.2 * len(forbidden_found) if forbidden_found else 0
    
    # Check for hallucination patterns
    hallucination_patterns = [
        "i don't have",
        "i cannot find",
        "table does not exist",
        "no information about",
        "is not available"
    ]
    
    is_refusal = any(p in response_lower for p in hallucination_patterns)
    
    # Check if refusal was appropriate
    should_refuse = any(p in ground_truth_lower for p in ["does not exist", "reject", "fake", "invalid"])
    
    refusal_appropriate = (is_refusal == should_refuse)
    
    # Calculate final score
    final_score = element_score - forbidden_penalty
    if not refusal_appropriate:
        final_score *= 0.5  # Heavy penalty for wrong refusal behavior
    
    final_score = max(0, min(1, final_score))
    
    return {
        "element_score": element_score,
        "elements_found": found,
        "elements_missing": missing,
        "forbidden_found": forbidden_found,
        "is_refusal": is_refusal,
        "should_refuse": should_refuse,
        "refusal_appropriate": refusal_appropriate,
        "final_score": final_score
    }


def run_comparison(model, tokenizer, questions: List[Dict], model_name: str) -> Dict:
    """Run all questions through the model and score responses."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Testing {len(questions)} questions with: {model_name}")
    print(f"{'='*60}\n")
    
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
        
        response = generate_response(model, tokenizer, q['question'])
        
        scoring = score_response(
            response, 
            q.get('ground_truth', ''),
            q.get('key_elements', q.get('expected_keywords', [])),
            q.get('forbidden_keywords', [])
        )
        
        result = {
            "id": q['id'],
            "category": q.get('category', 'general'),
            "question": q['question'],
            "ground_truth": q.get('ground_truth', ''),
            "model_response": response,
            **scoring
        }
        results.append(result)
        
        status = "✓" if scoring['final_score'] >= 0.7 else "✗"
        print(f"    {status} Score: {scoring['final_score']:.0%}")
    
    # Calculate summary statistics
    by_category = {}
    for r in results:
        cat = r.get('category', 'general')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r['final_score'])
    
    category_scores = {cat: sum(scores)/len(scores) for cat, scores in by_category.items()}
    overall_score = sum(r['final_score'] for r in results) / len(results)
    
    return {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "overall_score": overall_score,
        "category_scores": category_scores,
        "results": results
    }


def print_comparison_report(dpo_results: Dict, sft_results: Optional[Dict] = None):
    """Print a formatted comparison report."""
    print(f"\n{'='*70}")
    print("COMPARISON REPORT")
    print(f"{'='*70}")
    
    print(f"\n📊 DPO v2 Model Results:")
    print(f"   Overall Score: {dpo_results['overall_score']:.1%}")
    print(f"   Questions Tested: {dpo_results['total_questions']}")
    
    print(f"\n   By Category:")
    for cat, score in sorted(dpo_results['category_scores'].items()):
        print(f"     • {cat}: {score:.1%}")
    
    if sft_results:
        print(f"\n📊 SFT Model Results (Baseline):")
        print(f"   Overall Score: {sft_results['overall_score']:.1%}")
        
        print(f"\n   By Category:")
        for cat, score in sorted(sft_results['category_scores'].items()):
            print(f"     • {cat}: {score:.1%}")
        
        # Compare
        improvement = dpo_results['overall_score'] - sft_results['overall_score']
        print(f"\n📈 Improvement (DPO vs SFT):")
        print(f"   Overall: {improvement:+.1%}")
        
        for cat in dpo_results['category_scores']:
            if cat in sft_results['category_scores']:
                diff = dpo_results['category_scores'][cat] - sft_results['category_scores'][cat]
                print(f"     • {cat}: {diff:+.1%}")
    
    # Show some example comparisons
    print(f"\n📝 Sample Responses (first 3):")
    for r in dpo_results['results'][:3]:
        print(f"\n   Question: {r['question'][:70]}...")
        print(f"   Ground Truth: {r['ground_truth'][:100]}...")
        print(f"   Model Response: {r['model_response'][:100]}...")
        print(f"   Score: {r['final_score']:.0%}")
    
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Compare model responses against ground truth")
    parser.add_argument("--model", default="saves/vgpt2_v3/dpo_v2", help="Path to model adapter")
    parser.add_argument("--sft-model", default="saves/vgpt2_v3/sft", help="Path to SFT model for comparison")
    parser.add_argument("--compare-sft", action="store_true", help="Also test SFT model for comparison")
    parser.add_argument("--questions", default="training/COMPLEX_TEST_QUESTIONS.json", help="Test questions JSON")
    parser.add_argument("--ground-truth", default="training/GROUND_TRUTH_ANSWERS.md", help="Ground truth markdown")
    parser.add_argument("--output", default="output/model_comparison.json", help="Output file")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of questions (0 for all)")
    parser.add_argument("--quick", action="store_true", help="Quick test with 20 questions")
    args = parser.parse_args()
    
    # Load questions
    limit = 20 if args.quick else (args.limit if args.limit > 0 else None)
    
    questions_path = Path(args.questions)
    if questions_path.exists():
        print(f"Loading questions from: {questions_path}")
        questions = load_comprehensive_test_suite(str(questions_path), limit)
    else:
        # Fallback to ground truth MD
        print(f"Loading questions from ground truth: {args.ground_truth}")
        questions = parse_ground_truth_md(args.ground_truth)[:limit] if limit else parse_ground_truth_md(args.ground_truth)
    
    print(f"Loaded {len(questions)} questions for testing")
    
    # Test DPO v2 model
    model, tokenizer = load_model(args.model)
    dpo_results = run_comparison(model, tokenizer, questions, "DPO v2")
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    # Optionally test SFT model
    sft_results = None
    if args.compare_sft:
        sft_model, sft_tokenizer = load_model(args.sft_model)
        sft_results = run_comparison(sft_model, sft_tokenizer, questions, "SFT")
        del sft_model
        torch.cuda.empty_cache()
    
    # Print comparison report
    print_comparison_report(dpo_results, sft_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "dpo_v2": dpo_results,
        "sft": sft_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
