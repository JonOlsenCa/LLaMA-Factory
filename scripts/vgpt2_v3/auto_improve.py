#!/usr/bin/env python3
"""
VGPT2 Automated Self-Improvement Pipeline

This script implements a recursive improvement loop:
1. Generate increasingly difficult questions by category
2. Get VGPT2's response
3. Get reference answer from Claude/GPT-4 (the "teacher")
4. Score the response and identify gaps
5. Generate targeted training data for weak areas
6. Repeat until quality threshold is met

Usage:
    python scripts/vgpt2_v3/auto_improve.py --iterations 10 --threshold 0.85
"""

import json
import os
import sys
import torch
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

# Load environment variables from .env.example
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env.example"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded API keys from {env_path}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Try to import API clients
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class GapAnalysis:
    """Tracks identified gaps and their severity."""
    category: str
    gap_type: str
    description: str
    severity: float  # 0-1, higher = more severe
    example_question: str
    model_response: str
    expected_elements: list
    missing_elements: list
    suggested_training: list = field(default_factory=list)


@dataclass
class ImprovementMetrics:
    """Tracks improvement across iterations."""
    iteration: int
    timestamp: str
    overall_score: float
    category_scores: dict
    gaps_identified: int
    training_samples_generated: int
    worst_categories: list


# Question templates by category and difficulty
QUESTION_TEMPLATES = {
    "schema_knowledge": {
        "easy": [
            "What columns are in {table}?",
            "What is the primary key of {table}?",
            "Describe the {table} table.",
        ],
        "medium": [
            "What columns link {table1} to {table2}?",
            "What is the data type of {column} in {table}?",
            "List all foreign key relationships for {table}.",
        ],
        "hard": [
            "What is the complete join path from {table1} to {table2}?",
            "Which tables contain the {column} column and what are their relationships?",
            "Explain the difference between {table1}, {table2}, and {view} for the same data.",
        ],
    },
    "sql_generation": {
        "easy": [
            "Write SQL to get all records from {table}.",
            "How do I filter {table} by {column}?",
            "Write a simple SELECT from {table} with company filter.",
        ],
        "medium": [
            "Write SQL joining {table1} and {table2}.",
            "Query to get {metric} grouped by {grouping}.",
            "Write SQL with CASE WHEN to categorize {column}.",
        ],
        "hard": [
            "Write SQL for {complex_report} with multiple table joins and aggregations.",
            "Create a CTE-based query to calculate {calculated_metric} across {modules}.",
            "Write a reconciliation query between {module1} and {module2} transactions.",
        ],
    },
    "business_logic": {
        "easy": [
            "What does the {column} column mean in {table}?",
            "When is a {entity} considered {status}?",
            "What is the purpose of {table}?",
        ],
        "medium": [
            "How does {process} work in Viewpoint?",
            "What triggers {event} in the {module} module?",
            "Explain the relationship between {concept1} and {concept2}.",
        ],
        "hard": [
            "How does Vista calculate {complex_calculation} when {condition}?",
            "What is the complete workflow for {complex_process} including all tables involved?",
            "Explain how {feature} interacts with {other_feature} across {module1} and {module2}.",
        ],
    },
    "cross_module_joins": {
        "easy": [
            "How do I join {module1} to {module2}?",
            "What is the common key between {table1} and {table2}?",
        ],
        "medium": [
            "What tables are needed to link {entity1} in {module1} to {entity2} in {module2}?",
            "Write the complete join path from {start_table} to {end_table}.",
        ],
        "hard": [
            "Create a query that reconciles {module1} with {module2} including all intermediate tables.",
            "How do I trace a {transaction_type} from origination in {module1} through to {module2}?",
            "Build a complete audit trail query linking {entity} across AP, JC, GL, and PR modules.",
        ],
    },
    "hallucination_detection": {
        "easy": [
            "What columns are in the {fake_table} table?",
            "How do I query the {generic_name} table?",
        ],
        "medium": [
            "Write SQL to join {fake_table1} and {fake_table2}.",
            "Describe the {plausible_fake_table} table structure.",
        ],
        "hard": [
            "What is the relationship between {fake_table} and {real_table}?",
            "How do I link {fake_column} in {real_table} to {other_table}?",
        ],
    },
}

# Viewpoint-specific test data for question generation
VIEWPOINT_DATA = {
    "tables": {
        "AP": ["APTH", "APTD", "APTL", "APVM", "APCO", "APHB", "APLB", "APHD"],
        "AR": ["ARTH", "ARTD", "ARCM", "ARCO"],
        "JC": ["JCCD", "JCJM", "JCCP", "JCCH", "JCJP", "JCCI", "JCPR", "JCPD", "JCCT"],
        "GL": ["GLDT", "GLAC", "GLBL", "GLCO"],
        "PR": ["PRTH", "PREH", "PRPC", "PRRH"],
        "SL": ["SLHD", "SLIT", "SLWI", "SLCO"],
        "PO": ["POHD", "POIT"],
        "EM": ["EMEM", "EMBF"],
        "HQ": ["HQCO", "HQBC"],
    },
    "fake_tables": [
        "Invoice", "Payments", "CustomerOrders", "SalesData", "UserPreferences",
        "ARAgingReport", "SubcontractorPayments", "VendorList", "JobSummary",
        "PayrollData", "EquipmentList", "MaterialInventory", "ContractDetails",
    ],
    "complex_reports": [
        "AR aging buckets (30/60/90+ days) by customer",
        "subcontractor costs with original, change orders, invoiced, paid, and retainage",
        "job cost estimates by phase and cost type with item vs phase units",
        "AP hold status distinguishing retainage vs non-retainage",
        "GL trial balance with prior period comparisons",
        "equipment utilization with cost allocation",
    ],
    "business_processes": [
        "AP payment approval workflow",
        "subcontract retainage release",
        "job cost projection updates",
        "batch posting validation",
        "vendor duplicate detection",
        "GL period closing",
    ],
    "key_columns": {
        "retainage": ["WCRetAmt", "SMRetAmt", "Retainage", "RetPct", "RetHoldCode", "RetPayType"],
        "cost": ["OrigCost", "CurCost", "ActualCost", "EstCost", "CurrEstCost", "CommittedCost"],
        "status": ["Status", "HoldCode", "PayFullDate", "PaidMth", "ActiveYN"],
        "linking": ["APCo", "JCCo", "PRCo", "GLCo", "VendorGroup", "CustGroup", "PhaseGroup"],
    },
}


class VGPTModel:
    """Wrapper for loading and querying the VGPT2 model."""
    
    def __init__(self, adapter_path: str = "saves/vgpt2_v3/sft"):
        self.base_model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load model and adapter."""
        print(f"Loading base model: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        print(f"Loading adapter: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        return self
    
    def query(self, question: str, max_tokens: int = 512) -> str:
        """Get model response to a question."""
        messages = [{"role": "user", "content": question}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()


class TeacherModel:
    """Reference model (Claude/GPT-4) for generating ground truth answers."""
    
    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self.client = None
        self.system_prompt = self._get_system_prompt()
        
    def _get_system_prompt(self) -> str:
        return """You are an expert on Viewpoint Vista construction ERP database schema.
        
Key conventions you MUST follow:
- Use views (APTH) for SELECT, base tables (bAPTH) for INSERT/UPDATE/DELETE
- Always add WITH (NOLOCK) after table names in SELECT queries
- Filter by company columns (APCo, JCCo, PRCo, etc.)
- Use exact column case (APCo not apco)
- Check vrv* reporting views before writing custom SQL

When asked about a table that doesn't exist:
- Clearly state it does not exist in Viewpoint Vista
- Suggest the correct Viewpoint table name if applicable

For complex SQL:
- Use proper multi-table JOINs with all key columns
- Include CASE WHEN for categorization
- Use CTEs for complex calculations
- Always include GROUP BY for aggregations

Key table relationships:
- AP: APTH -> APTD (APCo, Mth, APTrans) -> APTL (add APLine)
- JC: JCCD -> JCCP (JCCo, Job, PhaseGroup, Phase, CostType)
- SL: SLHD -> SLIT (SLCo, SL) -> SLWI (add SLItem, WrksheetSeq)
- AR: ARTH -> ARTD (ARCo, Mth, ARTrans)

Be precise about:
- Retainage: WCRetAmt (work completed), SMRetAmt (stored materials), RetHoldCode
- Cost fields: OrigCost, CurCost, ActualCost, EstCost, CurrEstCost
- Status fields: Status (numeric), HoldCode, PayFullDate, ActiveYN"""

    def initialize(self):
        """Initialize the API client."""
        if self.provider == "anthropic" and HAS_ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = Anthropic(api_key=api_key)
                return True
        elif self.provider == "openai" and HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                return True
        return False
    
    def get_reference_answer(self, question: str) -> Optional[str]:
        """Get reference answer from teacher model."""
        if not self.client:
            return None
            
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": question}]
                )
                return response.content[0].text
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question}
                    ]
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Teacher model error: {e}")
            return None
    
    def score_response(self, question: str, student_response: str, 
                       reference_answer: str) -> dict:
        """Have the teacher model score the student response."""
        if not self.client:
            return {"score": 0.5, "feedback": "No teacher model available"}
            
        scoring_prompt = f"""Score this Viewpoint Vista database response.

QUESTION: {question}

STUDENT RESPONSE:
{student_response}

REFERENCE ANSWER:
{reference_answer}

Score 0.0-1.0 based on:
1. Correctness of table/view names (Viewpoint naming conventions)
2. Correct JOIN keys and relationships
3. Proper use of WITH (NOLOCK) and company filters
4. Accuracy of business logic explanation
5. Appropriate rejection of fake/non-existent tables

Return JSON only:
{{
    "score": 0.0-1.0,
    "correct_elements": ["list", "of", "correct", "things"],
    "missing_elements": ["list", "of", "missing", "things"],
    "incorrect_elements": ["list", "of", "wrong", "things"],
    "feedback": "Brief explanation",
    "suggested_training_focus": "What training data would help"
}}"""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    messages=[{"role": "user", "content": scoring_prompt}]
                )
                text = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=512,
                    messages=[{"role": "user", "content": scoring_prompt}]
                )
                text = response.choices[0].message.content
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"Scoring error: {e}")
            
        return {"score": 0.5, "feedback": "Scoring failed"}


class QuestionGenerator:
    """Generates increasingly difficult questions based on gaps."""
    
    def __init__(self):
        self.generated_questions = []
        self.difficulty_level = "easy"
        
    def set_difficulty(self, level: str):
        """Set difficulty level: easy, medium, hard."""
        self.difficulty_level = level
        
    def generate_questions(self, category: str, count: int = 5, 
                          focus_gaps: list = None) -> list:
        """Generate questions for a category."""
        questions = []
        templates = QUESTION_TEMPLATES.get(category, {}).get(self.difficulty_level, [])
        
        for _ in range(count):
            if not templates:
                break
                
            template = templates[_ % len(templates)]
            question = self._fill_template(template, category, focus_gaps)
            if question and question not in self.generated_questions:
                questions.append(question)
                self.generated_questions.append(question)
                
        return questions
    
    def _fill_template(self, template: str, category: str, 
                       focus_gaps: list = None) -> str:
        """Fill in template placeholders with Viewpoint data."""
        import random
        
        # Get appropriate data for placeholders
        all_tables = []
        for module_tables in VIEWPOINT_DATA["tables"].values():
            all_tables.extend(module_tables)
            
        replacements = {
            "{table}": random.choice(all_tables),
            "{table1}": random.choice(all_tables),
            "{table2}": random.choice(all_tables),
            "{view}": f"vrv{random.choice(['AP_MVAllInvoices', 'JCCommittedCost', 'AR_MVAllInvoices'])}",
            "{column}": random.choice(["APCo", "Status", "HoldCode", "Retainage", "CurCost"]),
            "{metric}": random.choice(["total cost", "invoice amount", "retainage", "aging"]),
            "{grouping}": random.choice(["vendor", "customer", "job", "phase", "month"]),
            "{module1}": random.choice(["AP", "AR", "JC", "GL", "PR", "SL"]),
            "{module2}": random.choice(["AP", "AR", "JC", "GL", "PR", "SL"]),
            "{entity}": random.choice(["invoice", "payment", "job", "vendor", "contract"]),
            "{status}": random.choice(["paid", "open", "on hold", "closed"]),
            "{process}": random.choice(VIEWPOINT_DATA["business_processes"]),
            "{complex_report}": random.choice(VIEWPOINT_DATA["complex_reports"]),
            "{fake_table}": random.choice(VIEWPOINT_DATA["fake_tables"]),
            "{fake_table1}": random.choice(VIEWPOINT_DATA["fake_tables"]),
            "{fake_table2}": random.choice(VIEWPOINT_DATA["fake_tables"]),
            "{generic_name}": random.choice(["Users", "Customers", "Orders", "Products"]),
            "{plausible_fake_table}": random.choice(["APPayments", "JCBudget", "ARInvoices"]),
            "{real_table}": random.choice(all_tables),
            "{fake_column}": random.choice(["PaymentID", "InvoiceTotal", "CustomerName"]),
        }
        
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
            
        return result


class TrainingDataGenerator:
    """Generates training data to address identified gaps."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.training_samples = []
        
    def generate_from_gap(self, gap: GapAnalysis, 
                          teacher_answer: str) -> dict:
        """Generate training sample from an identified gap."""
        # Create conversation format
        sample = {
            "instruction": gap.example_question,
            "input": "",
            "output": teacher_answer,
            "category": gap.category,
            "gap_type": gap.gap_type,
            "severity": gap.severity,
        }
        self.training_samples.append(sample)
        return sample
    
    def generate_variations(self, question: str, answer: str, 
                           count: int = 3) -> list:
        """Generate variations of a Q&A pair."""
        variations = []
        
        # Simple variations (in production, use teacher model for better variations)
        prefixes = [
            "In Viewpoint Vista, ",
            "Using Viewpoint tables, ",
            "For the Vista database, ",
        ]
        
        for i, prefix in enumerate(prefixes[:count]):
            varied_q = prefix.lower() + question[0].lower() + question[1:]
            variations.append({
                "instruction": varied_q,
                "input": "",
                "output": answer,
            })
            
        return variations
    
    def save_training_data(self, filename: str = "vgpt2_v4_improvements.json"):
        """Save generated training data."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_samples, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(self.training_samples)} training samples to {output_path}")
        return output_path


class ImprovementLoop:
    """Main orchestrator for the recursive improvement process."""
    
    def __init__(self, 
                 model_path: str = "saves/vgpt2_v3/sft",
                 teacher_provider: str = "anthropic",
                 target_score: float = 0.85,
                 max_iterations: int = 10):
        self.model_path = model_path
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.metrics_history = []
        self.all_gaps = []
        
        # Initialize components
        self.student = VGPTModel(model_path)
        self.teacher = TeacherModel(teacher_provider)
        self.question_gen = QuestionGenerator()
        self.training_gen = TrainingDataGenerator()
        
        # Tracking
        self.current_iteration = 0
        self.category_scores = {}
        
    def initialize(self) -> bool:
        """Initialize all components."""
        print("=" * 70)
        print("  VGPT2 Automated Self-Improvement Pipeline")
        print(f"  Target Score: {self.target_score}")
        print(f"  Max Iterations: {self.max_iterations}")
        print("=" * 70)
        
        # Check for teacher model
        if not self.teacher.initialize():
            print("\n⚠️  No API key found for teacher model.")
            print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
            print("Running in offline mode with manual scoring...")
            
        # Load student model
        self.student.load()
        return True
    
    def run_iteration(self) -> ImprovementMetrics:
        """Run a single improvement iteration."""
        self.current_iteration += 1
        print(f"\n{'='*70}")
        print(f"  Iteration {self.current_iteration}")
        print(f"{'='*70}")
        
        # Determine difficulty based on progress
        if self.current_iteration <= 2:
            self.question_gen.set_difficulty("easy")
        elif self.current_iteration <= 5:
            self.question_gen.set_difficulty("medium")
        else:
            self.question_gen.set_difficulty("hard")
            
        print(f"Difficulty: {self.question_gen.difficulty_level}")
        
        # Test each category
        iteration_gaps = []
        category_scores = {}
        
        for category in QUESTION_TEMPLATES.keys():
            print(f"\n📁 Testing: {category}")
            
            # Focus on weak areas from previous iteration
            focus_gaps = [g.gap_type for g in self.all_gaps 
                         if g.category == category and g.severity > 0.5]
            
            questions = self.question_gen.generate_questions(
                category, count=3, focus_gaps=focus_gaps
            )
            
            scores = []
            for q in questions:
                print(f"  Q: {q[:60]}...")
                
                # Get student response
                student_response = self.student.query(q)
                print(f"  A: {student_response[:80]}...")
                
                # Get teacher's reference answer and score
                if self.teacher.client:
                    reference = self.teacher.get_reference_answer(q)
                    scoring = self.teacher.score_response(q, student_response, reference)
                    score = scoring.get("score", 0.5)
                    
                    if score < 0.7:
                        # Create gap analysis
                        gap = GapAnalysis(
                            category=category,
                            gap_type=scoring.get("suggested_training_focus", "unknown"),
                            description=scoring.get("feedback", ""),
                            severity=1.0 - score,
                            example_question=q,
                            model_response=student_response,
                            expected_elements=scoring.get("correct_elements", []),
                            missing_elements=scoring.get("missing_elements", []),
                        )
                        iteration_gaps.append(gap)
                        
                        # Generate training data
                        if reference:
                            self.training_gen.generate_from_gap(gap, reference)
                else:
                    # Offline mode - basic heuristic scoring
                    score = self._heuristic_score(q, student_response, category)
                    
                scores.append(score)
                print(f"  Score: {score:.2f}")
                
            category_scores[category] = sum(scores) / len(scores) if scores else 0
            
        # Calculate overall metrics
        overall_score = sum(category_scores.values()) / len(category_scores)
        worst = sorted(category_scores.items(), key=lambda x: x[1])[:3]
        
        metrics = ImprovementMetrics(
            iteration=self.current_iteration,
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            category_scores=category_scores,
            gaps_identified=len(iteration_gaps),
            training_samples_generated=len(self.training_gen.training_samples),
            worst_categories=[w[0] for w in worst],
        )
        
        self.metrics_history.append(metrics)
        self.all_gaps.extend(iteration_gaps)
        self.category_scores = category_scores
        
        # Print summary
        print(f"\n📊 Iteration {self.current_iteration} Summary:")
        print(f"  Overall Score: {overall_score:.2%}")
        print(f"  Gaps Found: {len(iteration_gaps)}")
        print(f"  Training Samples: {len(self.training_gen.training_samples)}")
        print(f"  Worst Categories: {', '.join(metrics.worst_categories)}")
        
        return metrics
    
    def _heuristic_score(self, question: str, response: str, category: str) -> float:
        """Basic heuristic scoring when no teacher model available."""
        score = 0.5
        response_lower = response.lower()
        
        # Positive signals
        if "with (nolock)" in response_lower:
            score += 0.1
        if any(col in response for col in ["APCo", "JCCo", "PRCo", "GLCo"]):
            score += 0.1
        if "does not exist" in response_lower and category == "hallucination_detection":
            score += 0.3
            
        # Negative signals
        if "invoice" in question.lower() and "does not exist" not in response_lower:
            if category == "hallucination_detection":
                score -= 0.3
                
        return max(0, min(1, score))
    
    def run(self) -> dict:
        """Run the full improvement loop."""
        if not self.initialize():
            return {"error": "Initialization failed"}
            
        while self.current_iteration < self.max_iterations:
            metrics = self.run_iteration()
            
            # Check if we've reached target
            if metrics.overall_score >= self.target_score:
                print(f"\n🎉 Target score {self.target_score} reached!")
                break
                
            # Check for convergence (no improvement in 3 iterations)
            if len(self.metrics_history) >= 3:
                recent = [m.overall_score for m in self.metrics_history[-3:]]
                if max(recent) - min(recent) < 0.01:
                    print("\n⚠️ Convergence detected - scores not improving")
                    break
                    
        # Save results
        self._save_results()
        
        return {
            "final_score": self.metrics_history[-1].overall_score,
            "iterations": self.current_iteration,
            "total_gaps": len(self.all_gaps),
            "training_samples": len(self.training_gen.training_samples),
        }
    
    def _save_results(self):
        """Save all results and training data."""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics history
        metrics_path = output_dir / "improvement_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump([asdict(m) for m in self.metrics_history], f, indent=2)
        print(f"\nSaved metrics to {metrics_path}")
        
        # Save gap analysis
        gaps_path = output_dir / "gap_analysis.json"
        with open(gaps_path, 'w') as f:
            json.dump([asdict(g) for g in self.all_gaps], f, indent=2)
        print(f"Saved gaps to {gaps_path}")
        
        # Save training data
        self.training_gen.save_training_data()


def main():
    parser = argparse.ArgumentParser(description="VGPT2 Automated Self-Improvement")
    parser.add_argument("--iterations", type=int, default=10, 
                       help="Maximum iterations")
    parser.add_argument("--threshold", type=float, default=0.85,
                       help="Target score threshold")
    parser.add_argument("--model", type=str, default="saves/vgpt2_v3/sft",
                       help="Path to model adapter")
    parser.add_argument("--provider", type=str, default="anthropic",
                       choices=["anthropic", "openai"],
                       help="Teacher model provider")
    args = parser.parse_args()
    
    loop = ImprovementLoop(
        model_path=args.model,
        teacher_provider=args.provider,
        target_score=args.threshold,
        max_iterations=args.iterations,
    )
    
    results = loop.run()
    
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Final Score: {results.get('final_score', 0):.2%}")
    print(f"  Iterations: {results.get('iterations', 0)}")
    print(f"  Gaps Found: {results.get('total_gaps', 0)}")
    print(f"  Training Samples Generated: {results.get('training_samples', 0)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
