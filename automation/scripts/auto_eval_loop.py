# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Automated question generation, grading, and gap-mining loop for VGPT2 models.

This script generates increasingly hard questions, queries the tuned model, grades
answers with an external LLM, and writes gap-focused results to JSONL. It expects
OPENAI_API_KEY (and optional OPENAI_BASE_URL) in a loaded .env.

Quality-first defaults (A6000, bf16, flash-attn assumed):
- Base model: Qwen/Qwen2.5-7B-Instruct
- Adapter: saves/vgpt2_v3/sft
- Hardness levels: 3, 4 questions per level per category
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CATEGORIES: Sequence[str] = (
    "complex_sql",
    "business_logic",
    "cross_module_join",
    "hallucination",
    "naming_conventions",
)


@dataclass
class QuestionItem:
    category: str
    hardness: int
    question: str
    rubric: Dict[str, Any]


@dataclass
class GradedItem:
    category: str
    hardness: int
    question: str
    rubric: Dict[str, Any]
    model_response: str
    grade: Dict[str, Any]


def load_env_file(env_path: Path) -> None:
    load_dotenv(dotenv_path=env_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"Missing OPENAI_API_KEY; set it in {env_path} or the environment.")


def build_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def load_model(base_model: str, adapter_path: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def _call_llm_json(client: OpenAI, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=700,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def generate_questions(
    client: OpenAI,
    model: str,
    hardness_levels: int,
    per_level: int,
) -> List[QuestionItem]:
    items: List[QuestionItem] = []
    for category in CATEGORIES:
        for hardness in range(1, hardness_levels + 1):
            prompt = (
                "You are a data generator for Viewpoint Vista SQL assistant training. "
                "Create questions and rubrics that expose mistakes in joins, company filtering, "
                "retainage logic, aging buckets, and hallucination rejection. Hardness "
                f"level {hardness} means more joins, more required columns, and more traps. "
                "Return JSON with fields: questions (array of strings) and rubrics (array of "
                "objects) where each rubric lists required_elements and reject_conditions."
            )
            payload = _call_llm_json(
                client,
                [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": f"category={category}\n{prompt}\ncount={per_level}"},
                ],
                model=model,
            )
            q_list: Iterable[str] = payload.get("questions", [])
            r_list: Iterable[Dict[str, Any]] = payload.get("rubrics", [])
            for q, r in zip(q_list, r_list):
                items.append(QuestionItem(category=category, hardness=hardness, question=q, rubric=r))
    return items


def run_model(model: Any, tokenizer: Any, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def grade_response(
    client: OpenAI,
    grader_model: str,
    question: str,
    rubric: Dict[str, Any],
    answer: str,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "Return strict JSON with fields: score (0-1), found, missing, notes."},
        {
            "role": "user",
            "content": (
                "Grade the assistant answer against the rubric. "
                "Rubric keys: required_elements (array of strings), reject_conditions (array of strings). "
                "Penalize if any reject_conditions are present or if required elements are missing."
            ),
        },
        {"role": "user", "content": f"QUESTION:\n{question}"},
        {"role": "user", "content": f"RUBRIC:\n{json.dumps(rubric)}"},
        {"role": "assistant", "content": f"ANSWER:\n{answer}"},
    ]
    return _call_llm_json(client, messages, model=grader_model)


def save_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto eval loop for VGPT2 models")
    parser.add_argument("--env_path", type=Path, default=Path(".env"))
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default="saves/vgpt2_v3/sft")
    parser.add_argument("--question_model", default="gpt-4o-mini")
    parser.add_argument("--grader_model", default="gpt-4o-mini")
    parser.add_argument("--hardness_levels", type=int, default=3)
    parser.add_argument("--per_level", type=int, default=4)
    parser.add_argument("--output_path", type=Path, default=Path("output/auto_eval_results.jsonl"))
    args = parser.parse_args()

    load_env_file(args.env_path)
    client = build_openai_client()
    model, tokenizer = load_model(args.base_model, args.adapter_path)

    questions = generate_questions(
        client=client,
        model=args.question_model,
        hardness_levels=args.hardness_levels,
        per_level=args.per_level,
    )

    graded: List[GradedItem] = []
    for item in questions:
        response = run_model(model, tokenizer, item.question)
        grade = grade_response(
            client=client,
            grader_model=args.grader_model,
            question=item.question,
            rubric=item.rubric,
            answer=response,
        )
        graded.append(
            GradedItem(
                category=item.category,
                hardness=item.hardness,
                question=item.question,
                rubric=item.rubric,
                model_response=response,
                grade=grade,
            )
        )

    records = [
        {
            "category": g.category,
            "hardness": g.hardness,
            "question": g.question,
            "rubric": g.rubric,
            "model_response": g.model_response,
            "grade": g.grade,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for g in graded
    ]
    save_jsonl(args.output_path, records)
    print(f"Saved results to {args.output_path} ({len(records)} items)")


if __name__ == "__main__":
    main()
