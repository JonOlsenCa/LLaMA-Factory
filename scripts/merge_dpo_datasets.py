#!/usr/bin/env python3
"""
Merge hallucination-focused DPO pairs with best existing pairs.

Creates a balanced dataset with heavy emphasis on hallucination rejection.
"""

import json
import random
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_nolock_pair(pair):
    """Check if pair teaches NOLOCK usage."""
    chosen = pair.get("chosen", "")
    rejected = pair.get("rejected", "")
    return "WITH (NOLOCK)" in chosen and "WITH (NOLOCK)" not in rejected


def is_join_pair(pair):
    """Check if pair teaches proper JOIN patterns."""
    instruction = pair.get("instruction", "").lower()
    return "join" in instruction


def is_case_sensitivity_pair(pair):
    """Check if pair teaches case sensitivity."""
    chosen = pair.get("chosen", "")
    return "Latin1_General_BIN" in chosen or "case-sensitive" in chosen.lower()


def extract_best_existing_pairs(existing_data, max_count=500):
    """Extract best NOLOCK, JOIN, and case sensitivity pairs."""
    
    nolock_pairs = []
    join_pairs = []
    case_pairs = []
    
    for pair in existing_data:
        if is_join_pair(pair):
            join_pairs.append(pair)
        elif is_case_sensitivity_pair(pair):
            case_pairs.append(pair)
        elif is_nolock_pair(pair):
            nolock_pairs.append(pair)
    
    print(f"Found in existing dataset:")
    print(f"  NOLOCK pairs: {len(nolock_pairs)}")
    print(f"  JOIN pairs: {len(join_pairs)}")
    print(f"  Case sensitivity pairs: {len(case_pairs)}")
    
    # Select balanced mix
    selected = []
    
    # Take ~200 JOIN pairs (most valuable for teaching relationships)
    random.shuffle(join_pairs)
    selected.extend(join_pairs[:200])
    
    # Take ~150 NOLOCK pairs  
    random.shuffle(nolock_pairs)
    selected.extend(nolock_pairs[:150])
    
    # Take ~100 case sensitivity pairs
    random.shuffle(case_pairs)
    selected.extend(case_pairs[:100])
    
    # Deduplicate by instruction
    seen = set()
    unique = []
    for pair in selected:
        key = pair.get("instruction", "")
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    
    print(f"\nSelected {len(unique)} unique pairs from existing dataset")
    return unique


def deduplicate_halluc_pairs(halluc_data):
    """Remove duplicate hallucination pairs."""
    seen = set()
    unique = []
    
    for pair in halluc_data:
        key = pair.get("instruction", "")
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    
    return unique


def main():
    print("=" * 60)
    print("Merging DPO Datasets")
    print("=" * 60)
    
    # Load hallucination pairs
    halluc_path = Path("data/vgpt2_v3_dpo_halluc_raw.json")
    halluc_data = load_json(halluc_path)
    print(f"\nLoaded hallucination pairs: {len(halluc_data)}")
    
    # Deduplicate
    halluc_data = deduplicate_halluc_pairs(halluc_data)
    print(f"After deduplication: {len(halluc_data)}")
    
    # Load existing DPO dataset
    existing_path = Path("data/vgpt2_v3_dpo.json")
    existing_data = load_json(existing_path)
    print(f"\nLoaded existing DPO pairs: {len(existing_data)}")
    
    # Extract best existing pairs
    best_existing = extract_best_existing_pairs(existing_data, max_count=500)
    
    # Combine datasets
    combined = halluc_data + best_existing
    
    # Shuffle
    random.seed(42)
    random.shuffle(combined)
    
    print(f"\n" + "=" * 60)
    print(f"Final Dataset Summary")
    print("=" * 60)
    print(f"Hallucination pairs: {len(halluc_data)}")
    print(f"Quality pairs (NOLOCK/JOIN/case): {len(best_existing)}")
    print(f"Total combined: {len(combined)}")
    
    # Calculate percentages
    halluc_pct = len(halluc_data) / len(combined) * 100
    quality_pct = len(best_existing) / len(combined) * 100
    print(f"\nDataset composition:")
    print(f"  Hallucination focus: {halluc_pct:.1f}%")
    print(f"  Quality/style focus: {quality_pct:.1f}%")
    
    # Save combined dataset
    output_path = Path("data/vgpt2_v3_dpo_v2.json")
    save_json(combined, output_path)
    print(f"\nSaved to: {output_path}")
    
    # Update dataset_info.json
    update_dataset_info(output_path.name)
    
    return combined


def update_dataset_info(filename):
    """Update dataset_info.json with new dataset."""
    info_path = Path("data/dataset_info.json")
    info = load_json(info_path)
    
    # Add new dataset entry
    info["vgpt2_v3_dpo_v2"] = {
        "file_name": filename,
        "formatting": "alpaca",
        "ranking": True,
        "columns": {
            "prompt": "instruction",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }
    
    save_json(info, info_path)
    print(f"Updated dataset_info.json")


if __name__ == "__main__":
    main()

