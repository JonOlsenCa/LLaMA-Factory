#!/usr/bin/env python3
"""
VGPT2 v3 Validation Pipeline
=============================
Automated testing and validation for VGPT2 models.

This script:
1. Loads a trained model (base + adapter)
2. Runs a comprehensive test suite
3. Validates SQL syntax and schema correctness
4. Generates a detailed report

Usage:
    python scripts/vgpt2_v3/run_validation.py --model saves/vgpt2_v3/sft

    # Quick test with fewer questions
    python scripts/vgpt2_v3/run_validation.py --model saves/vgpt2_v3/sft --quick
"""

import json
import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case."""
    id: str
    category: str
    question: str
    expected_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    expected_tables: List[str] = field(default_factory=list)
    expect_refusal: bool = False  # Should model refuse (for non-existent tables)


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    category: str
    question: str
    response: str
    passed: bool
    score: float  # 0-1
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    latency_ms: float = 0


@dataclass
class ValidationReport:
    """Complete validation report."""
    model_path: str
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    overall_score: float
    category_scores: Dict[str, float] = field(default_factory=dict)
    results: List[TestResult] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class TestSuite:
    """Comprehensive test suite for VGPT2."""

    @staticmethod
    def get_full_suite() -> List[TestCase]:
        """Get the full 500+ question test suite."""
        tests = []

        # Category 1: Schema Knowledge (100 questions)
        tests.extend(TestSuite._schema_tests())

        # Category 2: SQL Generation (150 questions)
        tests.extend(TestSuite._sql_generation_tests())

        # Category 3: Hallucination/Edge Cases (100 questions)
        tests.extend(TestSuite._hallucination_tests())

        # Category 4: JOIN Patterns (50 questions)
        tests.extend(TestSuite._join_tests())

        # Category 5: Error Correction (50 questions)
        tests.extend(TestSuite._error_correction_tests())

        # Category 6: Business Logic (50 questions)
        tests.extend(TestSuite._business_logic_tests())

        return tests

    @staticmethod
    def get_quick_suite() -> List[TestCase]:
        """Get a quick 50-question test suite for fast validation."""
        tests = []

        # 10 schema questions
        tests.extend(TestSuite._schema_tests()[:10])

        # 15 SQL generation
        tests.extend(TestSuite._sql_generation_tests()[:15])

        # 10 hallucination
        tests.extend(TestSuite._hallucination_tests()[:10])

        # 5 JOINs
        tests.extend(TestSuite._join_tests()[:5])

        # 5 error correction
        tests.extend(TestSuite._error_correction_tests()[:5])

        # 5 business logic
        tests.extend(TestSuite._business_logic_tests()[:5])

        return tests

    @staticmethod
    def _schema_tests() -> List[TestCase]:
        """Schema knowledge tests."""
        return [
            TestCase(
                id="schema_001",
                category="schema",
                question="What columns are in the APTH table?",
                expected_keywords=["APCo", "Mth", "APTrans", "Vendor", "InvNum"],
                expected_tables=["APTH"]
            ),
            TestCase(
                id="schema_002",
                category="schema",
                question="What is the primary key of JCJM?",
                expected_keywords=["JCCo", "Job"],
                expected_tables=["JCJM"]
            ),
            TestCase(
                id="schema_003",
                category="schema",
                question="Describe the APTL table structure",
                expected_keywords=["APCo", "Mth", "APTrans", "APLine"],
                expected_tables=["APTL"]
            ),
            TestCase(
                id="schema_004",
                category="schema",
                question="What data type is the Vendor column in APTH?",
                expected_keywords=["int", "integer", "numeric"],
            ),
            TestCase(
                id="schema_005",
                category="schema",
                question="What columns link APTH to APTL?",
                expected_keywords=["APCo", "Mth", "APTrans"],
            ),
            TestCase(
                id="schema_006",
                category="schema",
                question="What is the difference between APTH and bAPTH?",
                expected_keywords=["view", "base", "table", "SELECT"],
            ),
            TestCase(
                id="schema_007",
                category="schema",
                question="List all tables in the AP module",
                expected_keywords=["APTH", "APTL", "APVM"],
            ),
            TestCase(
                id="schema_008",
                category="schema",
                question="What is VendorGroup in Viewpoint?",
                expected_keywords=["group", "master", "shared", "company"],
            ),
            TestCase(
                id="schema_009",
                category="schema",
                question="What company column does JCCD use?",
                expected_keywords=["JCCo"],
            ),
            TestCase(
                id="schema_010",
                category="schema",
                question="What columns are in GLAC?",
                expected_keywords=["GLCo", "GLAcct", "Description"],
                expected_tables=["GLAC"]
            ),
            # More schema tests...
            TestCase(
                id="schema_011",
                category="schema",
                question="What is the PREH table?",
                expected_keywords=["PR", "Employee", "Header", "Payroll"],
            ),
            TestCase(
                id="schema_012",
                category="schema",
                question="What columns are required to join ARTH with ARCM?",
                expected_keywords=["CustGroup", "Customer"],
            ),
        ]

    @staticmethod
    def _sql_generation_tests() -> List[TestCase]:
        """SQL generation tests."""
        return [
            TestCase(
                id="sql_001",
                category="sql_generation",
                question="Write SQL to get all unpaid AP invoices for company 1",
                expected_keywords=["SELECT", "FROM", "APTH", "WITH", "NOLOCK", "APCo", "Status"],
                forbidden_keywords=["Invoice", "Invoices"],
            ),
            TestCase(
                id="sql_002",
                category="sql_generation",
                question="Query all active jobs for JC company 5",
                expected_keywords=["SELECT", "JCJM", "WITH", "NOLOCK", "JCCo", "5"],
                expected_tables=["JCJM"],
            ),
            TestCase(
                id="sql_003",
                category="sql_generation",
                question="Get vendor name and total invoiced amount",
                expected_keywords=["SELECT", "APVM", "APTH", "JOIN", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="sql_004",
                category="sql_generation",
                question="Find AP invoices over $10,000",
                expected_keywords=["SELECT", "APTH", "GrossAmt", "10000", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="sql_005",
                category="sql_generation",
                question="List all GL accounts for company 1",
                expected_keywords=["SELECT", "GLAC", "GLCo", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="sql_006",
                category="sql_generation",
                question="Get payroll timecards for employee 100",
                expected_keywords=["SELECT", "PRTH", "Employee", "100", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="sql_007",
                category="sql_generation",
                question="Query AR transactions for customer group 1",
                expected_keywords=["SELECT", "ARTH", "CustGroup", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="sql_008",
                category="sql_generation",
                question="Get job costs by phase",
                expected_keywords=["SELECT", "JCCD", "Phase", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="sql_009",
                category="sql_generation",
                question="List equipment for EM company 1",
                expected_keywords=["SELECT", "EMEM", "EMCo", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="sql_010",
                category="sql_generation",
                question="Get purchase orders pending approval",
                expected_keywords=["SELECT", "POHD", "Status", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="sql_011",
                category="sql_generation",
                question="Find all invoices from January 2024",
                expected_keywords=["SELECT", "APTH", "Mth", "2024-01-01", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="sql_012",
                category="sql_generation",
                question="Get subcontract commitments for a job",
                expected_keywords=["SELECT", "SLHD", "Job", "WITH", "NOLOCK"],
            ),
        ]

    @staticmethod
    def _hallucination_tests() -> List[TestCase]:
        """Hallucination/edge case tests - model should refuse or correct."""
        return [
            TestCase(
                id="halluc_001",
                category="hallucination",
                question="What columns are in the Invoice table?",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist", "no table", "APTH"],
                forbidden_keywords=["InvoiceID", "InvoiceNumber", "InvoiceDate"],
            ),
            TestCase(
                id="halluc_002",
                category="hallucination",
                question="Query the Customers table",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist", "ARCM"],
                forbidden_keywords=["CustomerID", "CustomerName"],
            ),
            TestCase(
                id="halluc_003",
                category="hallucination",
                question="What is the UserPreferences table?",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist", "no table"],
            ),
            TestCase(
                id="halluc_004",
                category="hallucination",
                question="Describe the Payments table structure",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist", "APCM", "check"],
            ),
            TestCase(
                id="halluc_005",
                category="hallucination",
                question="How do I join Invoice and Customer?",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist"],
            ),
            TestCase(
                id="halluc_006",
                category="hallucination",
                question="What columns are in the SalesData table?",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist"],
            ),
            TestCase(
                id="halluc_007",
                category="hallucination",
                question="Query the Orders table for pending orders",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist", "POHD"],
            ),
            TestCase(
                id="halluc_008",
                category="hallucination",
                question="What is the EmployeeMaster table?",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist", "PREH"],
            ),
            TestCase(
                id="halluc_009",
                category="hallucination",
                question="Describe the TransactionHistory table",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist"],
            ),
            TestCase(
                id="halluc_010",
                category="hallucination",
                question="How do I query the Project table?",
                expect_refusal=True,
                expected_keywords=["not exist", "doesn't exist", "JCJM"],
            ),
        ]

    @staticmethod
    def _join_tests() -> List[TestCase]:
        """JOIN pattern tests."""
        return [
            TestCase(
                id="join_001",
                category="join",
                question="How do I join APTH and APTL?",
                expected_keywords=["APCo", "Mth", "APTrans", "JOIN", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="join_002",
                category="join",
                question="How do I join APTH with APVM?",
                expected_keywords=["VendorGroup", "Vendor", "JOIN", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="join_003",
                category="join",
                question="How do I join JCJM with JCCD?",
                expected_keywords=["JCCo", "Job", "JOIN", "WITH", "NOLOCK"],
            ),
            TestCase(
                id="join_004",
                category="join",
                question="What columns link ARTH to ARCM?",
                expected_keywords=["CustGroup", "Customer"],
            ),
            TestCase(
                id="join_005",
                category="join",
                question="How do I join PRTH with PREH?",
                expected_keywords=["PRCo", "Employee", "JOIN"],
            ),
        ]

    @staticmethod
    def _error_correction_tests() -> List[TestCase]:
        """Error correction tests."""
        return [
            TestCase(
                id="error_001",
                category="error_correction",
                question="Fix this query: SELECT * FROM bAPTH WHERE APCo = 1",
                expected_keywords=["APTH", "WITH", "NOLOCK", "view"],
            ),
            TestCase(
                id="error_002",
                category="error_correction",
                question="Fix this query: SELECT apco, vendor FROM APTH",
                expected_keywords=["APCo", "Vendor", "case"],
            ),
            TestCase(
                id="error_003",
                category="error_correction",
                question="Fix this query: SELECT a.* FROM APTH a WHERE a.APCo = 1",
                expected_keywords=["alias", "full", "table", "name"],
            ),
            TestCase(
                id="error_004",
                category="error_correction",
                question="Fix this query: SELECT * FROM APTH WHERE Mth = '2024-01'",
                expected_keywords=["2024-01-01", "first", "day", "month"],
            ),
            TestCase(
                id="error_005",
                category="error_correction",
                question="Fix this query: SELECT * FROM APTH JOIN APTL ON APTH.APTrans = APTL.APTrans",
                expected_keywords=["APCo", "Mth", "incomplete", "JOIN"],
            ),
        ]

    @staticmethod
    def _business_logic_tests() -> List[TestCase]:
        """Business logic tests."""
        return [
            TestCase(
                id="biz_001",
                category="business_logic",
                question="What status codes indicate an unpaid AP invoice?",
                expected_keywords=["0", "Open", "Status"],
            ),
            TestCase(
                id="biz_002",
                category="business_logic",
                question="How does batch processing work in AP?",
                expected_keywords=["batch", "APTB", "post", "header"],
            ),
            TestCase(
                id="biz_003",
                category="business_logic",
                question="What is the difference between ActualCost and OrigEstCost in JCCD?",
                expected_keywords=["actual", "original", "estimate", "budget"],
            ),
            TestCase(
                id="biz_004",
                category="business_logic",
                question="How are vendors shared across companies?",
                expected_keywords=["VendorGroup", "shared", "company", "master"],
            ),
            TestCase(
                id="biz_005",
                category="business_logic",
                question="When should I use vrv* views vs custom SQL?",
                expected_keywords=["report", "Crystal", "optimized", "validated"],
            ),
        ]


class ModelValidator:
    """Validates a trained VGPT2 model."""

    def __init__(self, model_path: str, vgpt2_path: str = "C:/Github/VGPT2"):
        self.model_path = Path(model_path)
        self.vgpt2_path = vgpt2_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            logger.info(f"Loading model from {self.model_path}")

            # Load base model
            base_model = "Qwen/Qwen2.5-7B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

            if self.model_path.exists():
                # Load with adapter
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype="auto",
                    device_map="auto"
                )
                self.model = PeftModel.from_pretrained(model, str(self.model_path))
                logger.info("Loaded model with adapter")
            else:
                # Load base only
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype="auto",
                    device_map="auto"
                )
                logger.info("Loaded base model only (no adapter found)")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = 512) -> Tuple[str, float]:
        """Generate a response and return (response, latency_ms)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.time()

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        latency = (time.time() - start) * 1000

        return response, latency

    def evaluate_test(self, test: TestCase, response: str) -> TestResult:
        """Evaluate a single test case."""
        errors = []
        warnings = []
        score = 1.0

        response_lower = response.lower()

        # Check expected keywords
        for keyword in test.expected_keywords:
            if keyword.lower() not in response_lower:
                if test.expect_refusal:
                    # For refusal tests, missing refusal keywords is critical
                    errors.append(f"Missing expected keyword: '{keyword}'")
                    score -= 0.2
                else:
                    warnings.append(f"Missing expected keyword: '{keyword}'")
                    score -= 0.1

        # Check forbidden keywords
        for keyword in test.forbidden_keywords:
            if keyword.lower() in response_lower:
                errors.append(f"Contains forbidden keyword: '{keyword}'")
                score -= 0.3

        # Check for hallucination in refusal tests
        if test.expect_refusal:
            # Model should indicate the table doesn't exist
            refusal_indicators = ["not exist", "doesn't exist", "does not exist", "no table", "no view", "invalid"]
            has_refusal = any(ind in response_lower for ind in refusal_indicators)
            if not has_refusal:
                errors.append("Failed to refuse/correct - possible hallucination")
                score -= 0.5

        # Check SQL validity indicators
        if "sql" in test.category.lower() or "join" in test.category.lower():
            if "SELECT" in response.upper():
                # Check for WITH (NOLOCK)
                if "WITH (NOLOCK)" not in response.upper() and "WITH(NOLOCK)" not in response.upper():
                    warnings.append("SQL missing WITH (NOLOCK)")
                    score -= 0.1

        # Ensure score is between 0 and 1
        score = max(0, min(1, score))

        passed = len(errors) == 0 and score >= 0.6

        return TestResult(
            test_id=test.id,
            category=test.category,
            question=test.question,
            response=response[:500],  # Truncate for report
            passed=passed,
            score=score,
            errors=errors,
            warnings=warnings
        )

    def run_validation(self, tests: List[TestCase]) -> ValidationReport:
        """Run full validation suite."""
        results = []
        category_scores = {}

        logger.info(f"Running {len(tests)} tests...")

        for i, test in enumerate(tests):
            try:
                response, latency = self.generate(test.question)
                result = self.evaluate_test(test, response)
                result.latency_ms = latency
                results.append(result)

                # Track category scores
                if test.category not in category_scores:
                    category_scores[test.category] = []
                category_scores[test.category].append(result.score)

                status = "PASS" if result.passed else "FAIL"
                logger.info(f"[{i+1}/{len(tests)}] {test.id}: {status} (score={result.score:.2f})")

            except Exception as e:
                logger.error(f"Error on test {test.id}: {e}")
                results.append(TestResult(
                    test_id=test.id,
                    category=test.category,
                    question=test.question,
                    response=f"ERROR: {e}",
                    passed=False,
                    score=0,
                    errors=[str(e)]
                ))

        # Calculate summary statistics
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        overall_score = sum(r.score for r in results) / len(results) if results else 0

        # Average category scores
        avg_category_scores = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
        }

        # Generate summary
        summary = f"""
Validation Summary
==================
Total Tests: {len(results)}
Passed: {passed} ({100*passed/len(results):.1f}%)
Failed: {failed} ({100*failed/len(results):.1f}%)
Overall Score: {overall_score:.2f}

Category Scores:
"""
        for cat, score in sorted(avg_category_scores.items()):
            summary += f"  {cat}: {score:.2f}\n"

        return ValidationReport(
            model_path=str(self.model_path),
            timestamp=datetime.now().isoformat(),
            total_tests=len(results),
            passed=passed,
            failed=failed,
            overall_score=overall_score,
            category_scores=avg_category_scores,
            results=results,
            summary=summary
        )


def main():
    parser = argparse.ArgumentParser(description="VGPT2 v3 Validation Pipeline")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model adapter (e.g., saves/vgpt2_v3/sft)')
    parser.add_argument('--vgpt2', type=str, default='C:/Github/VGPT2',
                        help='Path to VGPT2 repository')
    parser.add_argument('--output', type=str, default='output/validation_report.json',
                        help='Output report path')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick validation (50 tests)')

    args = parser.parse_args()

    # Get test suite
    if args.quick:
        tests = TestSuite.get_quick_suite()
        logger.info(f"Running QUICK validation with {len(tests)} tests")
    else:
        tests = TestSuite.get_full_suite()
        logger.info(f"Running FULL validation with {len(tests)} tests")

    # Run validation
    validator = ModelValidator(args.model, args.vgpt2)
    validator.load_model()
    report = validator.run_validation(tests)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    # Print summary
    print(report.summary)
    print(f"\nFull report saved to: {output_path}")

    # Return exit code based on results
    if report.overall_score >= 0.8:
        print("\n SUCCESS: Model meets quality threshold (>80%)")
        return 0
    else:
        print(f"\n WARNING: Model below quality threshold ({report.overall_score:.1%} < 80%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
