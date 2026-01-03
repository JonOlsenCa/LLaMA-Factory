# LLM Models Reference Guide

**Last Updated: 2025-12-12**

This comprehensive reference guide provides detailed information about the latest Large Language Models (LLMs) from major providers. This document is automatically generated from official API sources.

---

## Table of Contents

1. [Anthropic Claude Models](#anthropic-claude-models)
2. [Google Gemini Models](#google-gemini-models)
3. [xAI Grok Models](#xai-grok-models)
4. [OpenAI Models](#openai-models)
5. [How This Document is Generated](#how-this-document-is-generated)

---

## Anthropic Claude Models

### Current Claude Models (Most Recent First)

#### Claude Opus 4.5

**Model ID:** `claude-opus-4-5-20251101`  
**Alias:** `claude-opus-4-5-latest`  
**Released:** 2025-11-01  
**Pricing:** $15 / $75 per MTok (input/output)  
**Context:** 200K tokens  

#### Claude Haiku 4.5

**Model ID:** `claude-haiku-4-5-20251001`  
**Alias:** `claude-haiku-4-5-latest`  
**Released:** 2025-10-01  
**Pricing:** $0.8 / $4 per MTok (input/output)  
**Context:** 200K tokens  

#### Claude Sonnet 4.5

**Model ID:** `claude-sonnet-4-5-20250929`  
**Alias:** `claude-sonnet-4-5-latest`  
**Released:** 2025-09-29  
**Pricing:** $3 / $15 per MTok (input/output)  
**Context:** 200K tokens  

#### Claude Opus 4

**Model ID:** `claude-opus-4-20250514`  
**Alias:** `claude-opus-4-latest`  
**Released:** 2025-05-14  
**Pricing:** $15 / $75 per MTok (input/output)  
**Context:** 200K tokens  

#### Claude Sonnet 4

**Model ID:** `claude-sonnet-4-20250514`  
**Alias:** `claude-sonnet-4-latest`  
**Released:** 2025-05-14  
**Pricing:** $3 / $15 per MTok (input/output)  
**Context:** 200K tokens  

**Source:** https://docs.anthropic.com/en/docs/about-claude/models  

---

## Google Gemini Models

### Current Gemini Models (Most Recent First)

#### Gemini 3 Pro

**Released:** 2025-11  
**Status:** Preview  
**Context:** 1M tokens  
**Model ID:** `gemini-3-pro`  

#### Gemini 2.5 Pro

**Released:** 2025-06  
**Status:** Stable  
**Context:** 1M tokens  
**Model ID:** `gemini-2.5-pro`  

#### Gemini 2.5 Flash

**Released:** 2025-07  
**Status:** Stable  
**Context:** 1M tokens  
**Model ID:** `gemini-2.5-flash`  

#### Gemini 2.5 Flash Lite

**Released:** 2025-07  
**Status:** Stable  
**Context:** 1M tokens  
**Model ID:** `gemini-2.5-flash-lite`  

#### Gemini 2.0 Flash

**Released:** 2024-12  
**Status:** Stable  
**Context:** 1M tokens  
**Model ID:** `gemini-2.0-flash`  

#### Gemini 2.0 Flash Lite

**Released:** 2024-12  
**Status:** Stable  
**Context:** 1M tokens  
**Model ID:** `gemini-2.0-flash-lite`  

**Source:** https://ai.google.dev/gemini-api/docs/models  

---

## xAI Grok Models

### Current Grok Models

> ⚠️ **Manual Update Required:** xAI docs are blocked by Cloudflare.
> To update, provide screenshot from: https://docs.x.ai/docs/models#model-pricing

| Model ID | Context | Rate Limit | Pricing ($/1M tok) |
|----------|---------|------------|--------------------|
| `grok-4-1-fast-reasoning` | 2,000,000 | 4M tpm / 480 rpm | in: $0.20 / out: $0.50 |
| `grok-4-1-fast-non-reasoning` | 2,000,000 | 4M tpm / 480 rpm | in: $0.20 / out: $0.50 |
| `grok-code-fast-1` | 256,000 | 2M tpm / 480 rpm | in: $0.60 / out: $1.50 |
| `grok-4-fast-reasoning` | 2,000,000 | 4M tpm / 480 rpm | in: $0.20 / out: $0.50 |
| `grok-4-fast-non-reasoning` | 2,000,000 | 4M tpm / 480 rpm | in: $0.20 / out: $0.50 |
| `grok-4-0709` | 256,000 | 2M tpm / 480 rpm | in: $3.00 / out: $15.00 |
| `grok-3-mini` | 131,072 | 480 rpm | in: $0.30 / out: $0.50 |
| `grok-3` | 131,072 | 600 rpm | in: $3.00 / out: $15.00 |
| `grok-2-vision-1212` | 32,768 | 600 rpm | in: $2.00 / out: $10.00 |
| `grok-2-image-1212` | N/A | 300 rpm | in: $0.07 / out: per image |

> ⚠️ **Deprecated (DO NOT USE):** grok-2-1212, grok-2-latest, grok-vision-beta, grok-beta  

**Source:** https://docs.x.ai/docs/models#model-pricing (manual screenshot required)  
**Last Verified:** 2025-12-12  

---

## OpenAI Models

### Current OpenAI Models (Most Recent First)

> ⚠️ **API Note:** GPT-5.x/o3/o1/o4 models use `max_completion_tokens` (NOT `max_tokens`)

#### GPT-5.2

**Released:** 2025-12-11  
**Description:** Best model for coding and agentic tasks, improved safety and reduced hallucinations  

| Model ID | Alias | Context | Max Output | Knowledge Cutoff | Pricing ($/1M tok) | API |
|----------|-------|---------|------------|------------------|-------------------|-----|
| `gpt-5.2` | `gpt-5.2-2025-12-11` | 400,000 | 128,000 | Aug 31, 2025 | in: $1.75 / out: $14.00 | Chat Completions, Responses |
| `gpt-5.2-pro` | `gpt-5.2-pro-2025-12-11` | 400,000 | 128,000 | Aug 31, 2025 | in: $21.00 / out: $168.00 | Responses API ONLY |

#### GPT-5.1

**Released:** 2025-11-13  
**Description:** Best model for coding and agentic tasks with configurable reasoning  

| Model ID | Alias | Context | Max Output | Knowledge Cutoff | Pricing ($/1M tok) | API |
|----------|-------|---------|------------|------------------|-------------------|-----|
| `gpt-5.1` | `gpt-5.1-2025-11-13` | 400,000 | 128,000 | Sep 30, 2024 | in: $1.25 / out: $10.00 | Chat Completions, Responses |

#### GPT-5

**Released:** 2025-08-07  
**Description:** Advanced reasoning model family  

| Model ID | Alias | Context | Max Output | Knowledge Cutoff | Pricing ($/1M tok) | API |
|----------|-------|---------|------------|------------------|-------------------|-----|
| `gpt-5` | `gpt-5-2025-08-07` | 400,000 | 128,000 | Sep 30, 2024 | in: $1.25 / out: $10.00 | Chat Completions, Responses |
| `gpt-5-mini` | `gpt-5-mini-2025-08-07` | 400,000 | 128,000 | May 31, 2024 | in: $0.25 / out: $2.00 | Chat Completions, Responses |
| `gpt-5-nano` | `gpt-5-nano-2025-08-07` | 400,000 | 128,000 | May 31, 2024 | in: $0.05 / out: $0.40 | Chat Completions, Responses |

#### GPT-4.1

**Released:** 2025-04-14  
**Description:** Major improvements on coding, instruction following, and long context  

| Model ID | Alias | Context | Max Output | Knowledge Cutoff | Pricing ($/1M tok) | API |
|----------|-------|---------|------------|------------------|-------------------|-----|
| `gpt-4.1` | `gpt-4.1-2025-04-14` | 1,047,576 | 32,768 | May 31, 2024 | in: $2.00 / out: $8.00 | Chat Completions, Responses |
| `gpt-4.1-mini` | `gpt-4.1-mini-2025-04-14` | 1,047,576 | 32,768 | May 31, 2024 | in: $0.40 / out: $1.60 | Chat Completions, Responses |
| `gpt-4.1-nano` | `gpt-4.1-nano-2025-04-14` | 1,047,576 | 32,768 | May 31, 2024 | in: $0.10 / out: $0.40 | Chat Completions, Responses |

#### o3/o4 Reasoning Models

**Released:** 2025-04-16  
**Description:** Advanced reasoning models with enhanced problem solving  

| Model ID | Alias | Context | Max Output | Knowledge Cutoff | Pricing ($/1M tok) | API |
|----------|-------|---------|------------|------------------|-------------------|-----|
| `o3` | `o3-2025-04-16` | 200,000 | 100,000 | May 31, 2024 | in: $10.00 / out: $40.00 | Chat Completions, Responses |
| `o3-mini` | `o3-mini-2025-01-31` | 200,000 | 100,000 | Oct 2023 | in: $1.10 / out: $4.40 | Chat Completions, Responses |
| `o4-mini` | `o4-mini-2025-04-16` | 200,000 | 100,000 | May 31, 2024 | in: $1.10 / out: $4.40 | Chat Completions, Responses |

#### o1 Reasoning Models

**Released:** 2024-12-17  
**Description:** Original reasoning models  

| Model ID | Alias | Context | Max Output | Knowledge Cutoff | Pricing ($/1M tok) | API |
|----------|-------|---------|------------|------------------|-------------------|-----|
| `o1` | `o1-2024-12-17` | 200,000 | 100,000 | Oct 2023 | in: $15.00 / out: $60.00 | Chat Completions, Responses |
| `o1-mini` | `o1-mini-2024-09-12` | 128,000 | 65,536 | Oct 2023 | in: $1.10 / out: $4.40 | Chat Completions |

#### GPT-4o Series

**Released:** 2024-05-13  
**Description:** Multimodal models with text and vision  

| Model ID | Alias | Context | Max Output | Knowledge Cutoff | Pricing ($/1M tok) | API |
|----------|-------|---------|------------|------------------|-------------------|-----|
| `gpt-4o` | `gpt-4o-2024-11-20` | 128,000 | 16,384 | Oct 2023 | in: $2.50 / out: $10.00 | Chat Completions |
| `gpt-4o-mini` | `gpt-4o-mini-2024-07-18` | 128,000 | 16,384 | Oct 2023 | in: $0.15 / out: $0.60 | Chat Completions |

> ⚠️ **Deprecated (DO NOT USE):** `gpt-5-pro`, `o1-preview`  

> ⚠️ **Not Yet Available:** `gpt-5.2-mini`  

**Source:** https://platform.openai.com/docs/models  

---

## How This Document is Generated

This document is automatically generated using the `update_llm_reference.py` script.

### Data Sources by Provider

#### ✅ OpenAI (Fully Automated)
- **API Access:** `GET https://api.openai.com/v1/models` - Lists all available models
- **Model Specs:** Curated from verified sources (see below)
- **Validation:** Models tested via Chat Completions and Responses APIs
- **Accessible Documentation:**
  - https://www.promptfoo.dev/docs/providers/openai/ (detailed specs, pricing)
  - https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/models-sold-directly-by-azure
- **Blocked by Cloudflare (manual verification needed):**
  - https://platform.openai.com/docs/models
  - https://openai.com/api/pricing/

#### ✅ Anthropic Claude (Fully Automated)
- **API Access:** `GET https://api.anthropic.com/v1/models`
- **Documentation:** https://docs.anthropic.com/en/docs/about-claude/models

#### ✅ Google Gemini (Fully Automated)
- **API Access:** `GET https://generativelanguage.googleapis.com/v1beta/models`
- **Documentation:** https://ai.google.dev/gemini-api/docs/models

#### ⚠️ xAI Grok (Partial - May Need Manual Updates)
- **API Access:** `GET https://api.x.ai/v1/models`
- **Blocked by Cloudflare (manual verification needed):**
  - https://docs.x.ai/docs/models
  - https://x.ai/api

### Running the Update Script

```bash
# Set your API keys
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export XAI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"

# Run the update script
python update_llm_reference.py
```

### Requirements

```bash
pip install requests python-dotenv
```

---

**Document Version:** 2025-12-12
**Generated:** 2025-12-12 17:45:11
**Script:** `update_llm_reference.py`
