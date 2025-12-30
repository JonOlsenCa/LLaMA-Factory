# VGPT2 Fine-Tuning Executive Summary

**Date:** December 2025  
**Author:** Jon Olsen  
**Status:** v3 Training In Progress

---

## The Problem

VGPT2 is an AI assistant for Viewpoint Vista, a construction ERP system. Generic large language models (GPT-4, Claude, etc.) fail at generating accurate SQL queries for Vista because they:

- **Hallucinate** table and column names that don't exist
- **Miss critical conventions** like `WITH (NOLOCK)` required for production queries
- **Don't understand** Vista's complex composite key relationships (Company + Job + Contract + Item)
- **Ignore business logic** embedded in the schema (e.g., Status='A' means Active)

## Our Approach

We built a domain-specific fine-tuned model by training on our actual Viewpoint documentation—8,000+ stored procedures, 2,000+ views, schema metadata, and real SQL examples from production Crystal Reports.

### Training Method

We use **LoRA (Low-Rank Adaptation)** fine-tuning, which modifies only ~4% of the model's parameters while preserving its general language capabilities. This is more efficient than full fine-tuning and prevents catastrophic forgetting.

**Base Model:** Qwen2.5-7B-Instruct (7.9 billion parameters)  
**Framework:** LLaMA-Factory (open-source training toolkit)  
**Hardware:** NVIDIA RTX A6000 (48GB VRAM), AMD Threadripper 7960X, 128GB RAM

---

## Version 1 Results

**Training Data:** ~24,000 records from Vista documentation  
**Training Time:** 6.5 hours  
**Method:** Single-stage supervised fine-tuning (SFT)

### What Worked
- Model learned schema structure and column types
- Correctly answered 64% of test questions
- Generated valid SQL syntax 87% of the time

### What Failed
- **67% hallucination rate on edge cases** (invented fake table names)
- Inconsistent `WITH (NOLOCK)` usage
- Poor performance on complex multi-table JOINs
- No mechanism to reject questions about non-existent objects

**Root Cause:** The training data taught what correct SQL looks like, but never taught the model what *incorrect* SQL looks like. Without negative examples, the model couldn't distinguish valid from invalid requests.

---

## Version 2/3 Changes

### Data Improvements
| Change | v1 | v3 |
|--------|-----|-----|
| Training Records | 24K | 68K+ |
| Negative Examples | 0 | 3,000+ |
| SQL Pattern Coverage | Limited | Comprehensive |
| NOLOCK Training | 1.5% of data | 15%+ |

### Training Method Improvements

**Three-Stage Pipeline:**
1. **SFT (Supervised Fine-Tuning):** Learn correct patterns from documentation
2. **DPO (Direct Preference Optimization):** Learn to prefer correct SQL over incorrect SQL using paired examples
3. **KTO (Kahneman-Tversky Optimization):** Binary feedback refinement for edge cases

### Expected Results
| Metric | v1 | v3 Target |
|--------|-----|-----------|
| Overall Accuracy | 64% | 90%+ |
| Hallucination Rate | 67% | <5% |
| SQL Syntax Valid | 87% | 99% |
| NOLOCK Compliance | Inconsistent | 99%+ |

---

## Resource Investment

| Resource | v1 | v3 |
|----------|-----|-----|
| Data Preparation | 2 days | 5 days |
| Training Time | 6.5 hours | 12-18 hours |
| Validation/Testing | 2 hours | 8 hours |
| Hardware Cost | $0 (own equipment) | $0 |
| Cloud Equivalent | ~$50-100 | ~$150-300 |

---

## Scaling to Frontier Models

### What It Would Take

To achieve GPT-4-level performance on Vista SQL, we would need:

**Option A: Fine-tune a 70B+ Model**
- Base model: Llama-3-70B or Qwen2.5-72B
- Training: Full fine-tune with DeepSpeed ZeRO-3
- Hardware: 8x A100 80GB GPUs (~$25/hour cloud)
- Training time: 48-72 hours
- **Estimated cost: $2,000-5,000**
- **Expected improvement:** 95%+ accuracy, near-zero hallucination

**Option B: Distill from GPT-4/Claude**
- Generate 500K+ synthetic training examples using GPT-4
- Train smaller model on distilled knowledge
- **Estimated cost: $5,000-10,000** (API costs for generation)
- **Expected improvement:** 90-95% accuracy

**Option C: Partner with AI Provider**
- Custom fine-tuning through OpenAI or Anthropic
- Dedicated model deployment
- **Estimated cost: $50,000-100,000+/year**
- **Expected improvement:** 98%+ accuracy, full support

### What We'd Gain

| Capability | Current (7B) | Frontier (70B+) |
|------------|--------------|-----------------|
| Complex reasoning | Limited | Excellent |
| Multi-step queries | Often fails | Reliable |
| Natural language understanding | Good | Excellent |
| Edge case handling | Poor | Good |
| Context window | 4K tokens | 32K-128K tokens |

### Recommendation

**For now:** Complete v3 training with current hardware. If 90%+ accuracy is achieved, the ROI is excellent at near-zero cost.

**If v3 falls short:** Consider a 32B model with QLoRA (fits on current hardware) before jumping to cloud infrastructure.

**For production scale:** Budget $5,000-10,000 for a one-time 70B fine-tune if Vista SQL generation becomes a core business capability.

---

## Summary

We're building a specialized AI that understands Viewpoint Vista's database schema at an expert level. By fine-tuning on our own documentation and adding mechanisms to reject invalid requests, we expect to transform a 64% accurate prototype into a 90%+ production-ready system—at minimal cost using hardware we already own.

*Current Status: v3 SFT training in progress (Day 3 of 7-day pipeline)*

