#!/usr/bin/env python3
"""
Pre-flight Training Validation
==============================
Validates all training prerequisites before running actual training.
Catches configuration, data, and environment issues BEFORE they cause failures.

Usage:
    python training/run/validate_training.py --config automation/configs/vgpt2_v3/stage2_dpo.yaml
    python training/run/validate_training.py --config automation/configs/vgpt2_v3/stage1_sft.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_pass(msg: str):
    print(f"  ✅ PASS: {msg}")


def print_fail(msg: str):
    print(f"  ❌ FAIL: {msg}")


def print_warn(msg: str):
    print(f"  ⚠️  WARN: {msg}")


def print_info(msg: str):
    print(f"  ℹ️  INFO: {msg}")


def validate_yaml_config(config_path: str) -> dict:
    """Validate YAML config file exists and is parseable."""
    print_header("1. CONFIG FILE VALIDATION")
    
    errors = []
    config = None
    
    if not os.path.exists(config_path):
        print_fail(f"Config file not found: {config_path}")
        errors.append("Config file missing")
        return None
    
    print_pass(f"Config file exists: {config_path}")
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print_pass("YAML syntax is valid")
    except Exception as e:
        print_fail(f"YAML parse error: {e}")
        errors.append(f"YAML error: {e}")
        return None
    
    # Check required fields
    required_fields = ['model_name_or_path', 'stage', 'output_dir', 'dataset']
    for field in required_fields:
        if field in config:
            print_pass(f"Required field '{field}' present: {config[field]}")
        else:
            print_fail(f"Missing required field: {field}")
            errors.append(f"Missing field: {field}")
    
    return config if not errors else None


def validate_dataset(config: dict) -> bool:
    """Validate dataset file exists and has correct format."""
    print_header("2. DATASET VALIDATION")
    
    errors = []
    dataset_name = config.get('dataset', '')
    stage = config.get('stage', '')
    
    # Find dataset file
    dataset_info_path = Path("data/dataset_info.json")
    if not dataset_info_path.exists():
        print_fail("data/dataset_info.json not found")
        return False
    
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    if dataset_name not in dataset_info:
        print_fail(f"Dataset '{dataset_name}' not found in dataset_info.json")
        return False
    
    print_pass(f"Dataset '{dataset_name}' registered in dataset_info.json")
    
    # Get actual file path
    ds_config = dataset_info[dataset_name]
    file_name = ds_config.get('file_name', f"{dataset_name}.json")
    file_path = Path("data") / file_name
    
    if not file_path.exists():
        print_fail(f"Dataset file not found: {file_path}")
        return False
    
    print_pass(f"Dataset file exists: {file_path}")
    
    # Load and validate format
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print_pass(f"JSON syntax valid, {len(data):,} examples loaded")
    except Exception as e:
        print_fail(f"JSON parse error: {e}")
        return False
    
    # Check format based on stage
    if len(data) == 0:
        print_fail("Dataset is empty!")
        return False
    
    sample = data[0]
    
    if stage == 'dpo':
        required = ['instruction', 'chosen', 'rejected']
        for field in required:
            if field in sample:
                print_pass(f"DPO field '{field}' present")
            else:
                print_fail(f"DPO field '{field}' missing from samples")
                errors.append(f"Missing DPO field: {field}")
    elif stage == 'sft':
        if 'instruction' in sample or 'messages' in sample:
            print_pass("SFT format detected (instruction or messages)")
        else:
            print_fail("SFT requires 'instruction' or 'messages' field")
            errors.append("Invalid SFT format")
    
    # Sample validation
    print_info(f"Sample keys: {list(sample.keys())}")

    return len(errors) == 0


def validate_adapter_path(config: dict) -> bool:
    """Validate adapter path exists if specified (for DPO/continued training)."""
    print_header("3. ADAPTER/CHECKPOINT VALIDATION")

    adapter_path = config.get('adapter_name_or_path')

    if not adapter_path:
        print_info("No adapter_name_or_path specified (training from base model)")
        return True

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        print_fail(f"Adapter directory not found: {adapter_path}")
        return False

    print_pass(f"Adapter directory exists: {adapter_path}")

    # Check for required adapter files
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"

    if adapter_file.exists():
        size_mb = adapter_file.stat().st_size / (1024 * 1024)
        print_pass(f"adapter_model.safetensors exists ({size_mb:.1f} MB)")
    else:
        print_fail("adapter_model.safetensors not found")
        return False

    if adapter_config.exists():
        print_pass("adapter_config.json exists")
    else:
        print_fail("adapter_config.json not found")
        return False

    return True


def validate_output_dir(config: dict) -> bool:
    """Validate output directory is writable."""
    print_header("4. OUTPUT DIRECTORY VALIDATION")

    output_dir = Path(config.get('output_dir', 'output'))

    # Create if not exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print_pass(f"Output directory ready: {output_dir}")
    except Exception as e:
        print_fail(f"Cannot create output directory: {e}")
        return False

    # Check for existing checkpoints
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if checkpoints:
        print_warn(f"Found {len(checkpoints)} existing checkpoints in output dir")
        resume = config.get('resume_from_checkpoint', False)
        if resume:
            print_info("resume_from_checkpoint is enabled - will continue training")
        else:
            print_warn("resume_from_checkpoint is FALSE - training will start fresh")

    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(output_dir)
    free_gb = free / (1024**3)
    print_info(f"Free disk space: {free_gb:.1f} GB")

    if free_gb < 50:
        print_warn(f"Low disk space! Recommend at least 50GB free")
    else:
        print_pass(f"Sufficient disk space available")

    return True


def validate_windows_compatibility(config: dict) -> bool:
    """Check for Windows-specific issues."""
    print_header("5. WINDOWS COMPATIBILITY CHECK")

    issues = []

    # Check preprocessing workers
    num_workers = config.get('preprocessing_num_workers', 0)
    if num_workers > 4:
        print_warn(f"preprocessing_num_workers={num_workers} may cause timeout on Windows")
        print_info("Recommend: preprocessing_num_workers: 1 or 4")
        issues.append("High preprocessing_num_workers")
    else:
        print_pass(f"preprocessing_num_workers={num_workers} is Windows-safe")

    # Check dataloader workers
    dl_workers = config.get('dataloader_num_workers', 0)
    if dl_workers > 0:
        print_warn(f"dataloader_num_workers={dl_workers} may cause issues on Windows")
        print_info("Recommend: dataloader_num_workers: 0")
        issues.append("Non-zero dataloader_num_workers")
    else:
        print_pass(f"dataloader_num_workers={dl_workers} is Windows-safe")

    return len(issues) == 0


def validate_gpu(config: dict) -> bool:
    """Validate GPU availability and memory."""
    print_header("6. GPU VALIDATION")

    try:
        import torch
        if not torch.cuda.is_available():
            print_fail("CUDA not available!")
            return False

        print_pass(f"CUDA available: {torch.cuda.is_available()}")

        device_count = torch.cuda.device_count()
        print_pass(f"GPU count: {device_count}")

        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            free_mem = (torch.cuda.get_device_properties(i).total_memory -
                       torch.cuda.memory_allocated(i)) / (1024**3)
            print_info(f"GPU {i}: {name} ({total_mem:.1f} GB total, ~{free_mem:.1f} GB free)")

            # Memory warnings based on config
            batch_size = config.get('per_device_train_batch_size', 1)
            lora_rank = config.get('lora_rank', 64)
            cutoff_len = config.get('cutoff_len', 4096)

            # Rough estimate
            estimated_mem = 15 + (batch_size * cutoff_len * 0.002) + (lora_rank * 0.02)
            print_info(f"Estimated VRAM needed: ~{estimated_mem:.0f} GB")

            if estimated_mem > total_mem * 0.9:
                print_warn("May run out of GPU memory - consider reducing batch_size or cutoff_len")
            else:
                print_pass("GPU memory should be sufficient")

        return True

    except ImportError:
        print_fail("PyTorch not installed!")
        return False
    except Exception as e:
        print_fail(f"GPU check error: {e}")
        return False


def validate_model_access(config: dict) -> bool:
    """Validate model can be accessed (cached or downloadable)."""
    print_header("7. MODEL ACCESS VALIDATION")

    model_name = config.get('model_name_or_path', '')

    # Check if it's a local path
    if os.path.exists(model_name):
        print_pass(f"Model is local path: {model_name}")
        return True

    # Check HuggingFace cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = cache_dir / model_cache_name

    if model_cache_path.exists():
        print_pass(f"Model cached locally: {model_name}")
        snapshots = list((model_cache_path / "snapshots").glob("*")) if (model_cache_path / "snapshots").exists() else []
        if snapshots:
            print_info(f"Found {len(snapshots)} snapshot(s)")
        return True
    else:
        print_warn(f"Model not in cache - will download: {model_name}")
        print_info("First run may take a while to download")
        return True  # Not a failure, just a warning


def run_dry_run(config: dict, config_path: str) -> bool:
    """Attempt to initialize training without actually running."""
    print_header("8. DRY RUN (Initialize Training)")

    try:
        # Import LlamaFactory components
        from llamafactory.hparams import get_train_args

        print_info("Parsing training arguments...")

        # Pass the config dict directly (not {"config": path})
        model_args, data_args, training_args, finetuning_args, generating_args = \
            get_train_args(config)

        print_pass("Training arguments parsed successfully")
        print_info(f"Stage: {finetuning_args.stage}")
        print_info(f"LoRA rank: {finetuning_args.lora_rank}")
        print_info(f"Learning rate: {training_args.learning_rate}")
        print_info(f"Epochs: {training_args.num_train_epochs}")
        print_info(f"Batch size: {training_args.per_device_train_batch_size}")
        print_info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")

        effective_batch = (training_args.per_device_train_batch_size *
                          training_args.gradient_accumulation_steps)
        print_info(f"Effective batch size: {effective_batch}")

        return True

    except Exception as e:
        print_fail(f"Dry run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate training configuration")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config YAML file')
    parser.add_argument('--skip-dry-run', action='store_true',
                        help='Skip the dry run initialization test')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  TRAINING PRE-FLIGHT VALIDATION")
    print("="*60)
    print(f"  Config: {args.config}")
    print("="*60)

    results = {}

    # 1. Validate config
    config = validate_yaml_config(args.config)
    results['config'] = config is not None

    if not config:
        print("\n❌ VALIDATION FAILED - Cannot proceed without valid config")
        sys.exit(1)

    # 2. Validate dataset
    results['dataset'] = validate_dataset(config)

    # 3. Validate adapter
    results['adapter'] = validate_adapter_path(config)

    # 4. Validate output directory
    results['output'] = validate_output_dir(config)

    # 5. Windows compatibility
    results['windows'] = validate_windows_compatibility(config)

    # 6. GPU validation
    results['gpu'] = validate_gpu(config)

    # 7. Model access
    results['model'] = validate_model_access(config)

    # 8. Dry run (optional)
    if not args.skip_dry_run:
        results['dry_run'] = run_dry_run(config, args.config)

    # Summary
    print_header("VALIDATION SUMMARY")

    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("  ✅ ALL CHECKS PASSED - Ready for training!")
    else:
        print("  ❌ SOME CHECKS FAILED - Fix issues before training")
    print("="*60 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

