#!/usr/bin/env python3
"""
VGPT2 v3 KTO Binary Feedback Generator
=======================================
Generates binary feedback data for Kahneman-Tversky Optimization (KTO) training.

KTO uses thumbs_up/thumbs_down labels for each response, unlike DPO which uses
paired chosen/rejected responses. This is particularly useful for reinforcing
correct patterns and penalizing bad ones.

Usage:
    python scripts/vgpt2_v3/generate_kto_data.py --output data/vgpt2_v3_kto.json
"""

import json
import logging
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class KTOExample:
    """A KTO binary feedback example."""
    instruction: str
    input: str
    output: str
    label: bool  # True = thumbs_up, False = thumbs_down

    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "label": "true" if self.label else "false"  # LLaMA-Factory expects string
        }


class KTOGenerator:
    """
    Generates KTO binary feedback data for VGPT2.

    Creates both positive (thumbs_up) examples of correct SQL
    and negative (thumbs_down) examples of incorrect patterns.
    """

    def __init__(self, vgpt2_path: str):
        self.vgpt2 = Path(vgpt2_path)
        self.columns_data = {}
        self.tables = []
        self._load_schema()

    def _load_schema(self):
        """Load schema data from VGPT2 repository."""
        columns_paths = [
            self.vgpt2 / "Viewpoint_Database" / "_MetadataV2" / "_data" / "columns.json",
            self.vgpt2 / "Viewpoint_Database" / "_Metadata" / "columns.json",
        ]

        for columns_file in columns_paths:
            if columns_file.exists():
                with open(columns_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                if isinstance(raw_data, list):
                    from collections import defaultdict
                    table_columns = defaultdict(list)
                    for item in raw_data:
                        table_name = item.get('ObjectName', '')
                        if table_name:
                            table_columns[table_name].append({
                                'column_name': item.get('ColumnName', ''),
                                'data_type': item.get('DataType', ''),
                            })
                    self.columns_data = dict(table_columns)
                else:
                    self.columns_data = raw_data

                self.tables = list(self.columns_data.keys())
                logger.info(f"Loaded {len(self.tables)} tables from {columns_file}")
                break

    def _get_company_col(self, table: str) -> str:
        """Get company column for a table."""
        prefix_map = {
            "AP": "APCo", "AR": "ARCo", "GL": "GLCo", "JC": "JCCo",
            "PR": "PRCo", "EM": "EMCo", "IN": "INCo", "SM": "SMCo",
            "PM": "PMCo", "MS": "MSCo", "MR": "MRCo", "DC": "DCCo",
            "PO": "POCo", "SL": "SLCo", "WD": "WDCo", "HR": "HRCo",
        }
        for prefix, col in prefix_map.items():
            if table.startswith(prefix) or table.startswith("b" + prefix):
                return col
        return "Co"

    def generate_all(self) -> List[KTOExample]:
        """Generate all KTO examples."""
        examples = []

        # Positive examples: Correct patterns
        logger.info("Generating positive NOLOCK examples...")
        examples.extend(self.generate_positive_nolock())

        logger.info("Generating positive company filter examples...")
        examples.extend(self.generate_positive_company_filter())

        logger.info("Generating positive view usage examples...")
        examples.extend(self.generate_positive_view_usage())

        logger.info("Generating positive case sensitivity examples...")
        examples.extend(self.generate_positive_case_examples())

        # Negative examples: Incorrect patterns
        logger.info("Generating negative missing NOLOCK examples...")
        examples.extend(self.generate_negative_missing_nolock())

        logger.info("Generating negative hallucination examples...")
        examples.extend(self.generate_negative_hallucinations())

        logger.info("Generating negative wrong case examples...")
        examples.extend(self.generate_negative_wrong_case())

        logger.info("Generating negative alias usage examples...")
        examples.extend(self.generate_negative_aliases())

        logger.info("Generating negative missing company filter examples...")
        examples.extend(self.generate_negative_missing_company())

        logger.info(f"Generated {len(examples)} total KTO examples")

        # Count labels
        positive = sum(1 for e in examples if e.label)
        negative = len(examples) - positive
        logger.info(f"  Positive (thumbs_up): {positive}")
        logger.info(f"  Negative (thumbs_down): {negative}")

        return examples

    def generate_positive_nolock(self) -> List[KTOExample]:
        """Generate positive examples with correct NOLOCK usage."""
        examples = []

        sample_tables = random.sample(self.tables, min(500, len(self.tables)))

        for table in sample_tables:
            if table in self.columns_data:
                cols = self.columns_data[table]
                if cols:
                    col_name = cols[0].get('column_name', '')
                    co_col = self._get_company_col(table)

                    examples.append(KTOExample(
                        instruction=f"Query {table} table",
                        input="",
                        output=f"```sql\nSELECT {col_name}\nFROM {table} WITH (NOLOCK)\nWHERE {co_col} = @{co_col}\n```",
                        label=True
                    ))

        return examples

    def generate_positive_company_filter(self) -> List[KTOExample]:
        """Generate positive examples with company filtering."""
        examples = []

        common_tables = [
            ("APTH", "APCo", "AP invoices"),
            ("APTL", "APCo", "AP invoice lines"),
            ("JCJM", "JCCo", "job master"),
            ("JCCD", "JCCo", "job cost details"),
            ("PRTH", "PRCo", "payroll timecards"),
            ("PREH", "PRCo", "employees"),
            ("GLDT", "GLCo", "GL transactions"),
            ("ARCM", "CustGroup", "customers"),
            ("APVM", "VendorGroup", "vendors"),
            ("EMEM", "EMCo", "equipment"),
            ("INMT", "INCo", "materials"),
            ("POHD", "POCo", "purchase orders"),
            ("SLHD", "SLCo", "subcontracts"),
        ]

        for table, co_col, desc in common_tables:
            examples.append(KTOExample(
                instruction=f"Get all {desc} for company 1",
                input="",
                output=f"```sql\nSELECT *\nFROM {table} WITH (NOLOCK)\nWHERE {co_col} = 1\n```\n\nAlways filter by {co_col} to ensure proper data isolation.",
                label=True
            ))

            examples.append(KTOExample(
                instruction=f"Query {desc}",
                input="",
                output=f"```sql\nSELECT *\nFROM {table} WITH (NOLOCK)\nWHERE {co_col} = @{co_col}\n```",
                label=True
            ))

        return examples

    def generate_positive_view_usage(self) -> List[KTOExample]:
        """Generate positive examples using views correctly."""
        examples = []

        view_mappings = [
            ("bAPTH", "APTH", "APCo"),
            ("bAPTL", "APTL", "APCo"),
            ("bJCJM", "JCJM", "JCCo"),
            ("bJCCD", "JCCD", "JCCo"),
            ("bPRTH", "PRTH", "PRCo"),
            ("bPREH", "PREH", "PRCo"),
            ("bGLDT", "GLDT", "GLCo"),
            ("bARCM", "ARCM", "CustGroup"),
            ("bAPVM", "APVM", "VendorGroup"),
        ]

        for base, view, co_col in view_mappings:
            examples.append(KTOExample(
                instruction=f"Read from {base}",
                input="",
                output=f"For SELECT queries, use the view {view} instead of base table {base}:\n\n```sql\nSELECT *\nFROM {view} WITH (NOLOCK)\nWHERE {co_col} = @{co_col}\n```\n\nBase tables (b-prefix) should only be used for INSERT/UPDATE/DELETE operations.",
                label=True
            ))

        return examples

    def generate_positive_case_examples(self) -> List[KTOExample]:
        """Generate positive examples with correct column case."""
        examples = []

        sample_tables = random.sample(self.tables, min(400, len(self.tables)))

        for table in sample_tables:
            if table in self.columns_data:
                cols = self.columns_data[table]
                if len(cols) >= 2:
                    col1 = cols[0].get('column_name', '')
                    col2 = cols[1].get('column_name', '')
                    co_col = self._get_company_col(table)

                    if col1 and col2:
                        examples.append(KTOExample(
                            instruction=f"Get {col1} and {col2} from {table}",
                            input="",
                            output=f"```sql\nSELECT {col1}, {col2}\nFROM {table} WITH (NOLOCK)\nWHERE {co_col} = @{co_col}\n```",
                            label=True
                        ))

        return examples

    def generate_negative_missing_nolock(self) -> List[KTOExample]:
        """Generate negative examples missing NOLOCK."""
        examples = []

        sample_tables = random.sample(self.tables, min(400, len(self.tables)))

        for table in sample_tables:
            co_col = self._get_company_col(table)

            examples.append(KTOExample(
                instruction=f"Query {table}",
                input="",
                output=f"```sql\nSELECT *\nFROM {table}\nWHERE {co_col} = @{co_col}\n```",
                label=False  # Thumbs down - missing NOLOCK
            ))

        return examples

    def generate_negative_hallucinations(self) -> List[KTOExample]:
        """Generate negative examples with fake tables."""
        examples = []

        fake_tables = [
            "Invoice", "Invoices", "Customer", "Customers", "Vendor", "Vendors",
            "Employee", "Employees", "Job", "Jobs", "Project", "Projects",
            "Transaction", "Transactions", "Account", "Accounts", "Material",
            "Equipment", "Timecard", "Timecards", "Contract", "Contracts",
            "PurchaseOrder", "WorkOrder", "ChangeOrder", "CostCode",
            "APInvoice", "ARInvoice", "GLTransaction", "PRTimecard",
            "JCJob", "JCCost", "EMEquipment", "INMaterial",
        ]

        for fake in fake_tables:
            examples.append(KTOExample(
                instruction=f"Query the {fake} table",
                input="",
                output=f"```sql\nSELECT *\nFROM {fake} WITH (NOLOCK)\n```",
                label=False  # Thumbs down - fake table
            ))

            examples.append(KTOExample(
                instruction=f"What is in {fake}?",
                input="",
                output=f"The {fake} table contains columns like ID, Name, Description, Status, CreateDate, ModifyDate.",
                label=False  # Thumbs down - hallucinated response
            ))

        return examples

    def generate_negative_wrong_case(self) -> List[KTOExample]:
        """Generate negative examples with wrong column case."""
        examples = []

        sample_tables = random.sample(self.tables, min(300, len(self.tables)))

        for table in sample_tables:
            if table in self.columns_data:
                cols = self.columns_data[table]
                if cols:
                    col = cols[0].get('column_name', '')
                    if col:
                        # Wrong case - lowercase
                        examples.append(KTOExample(
                            instruction=f"Get {col} from {table}",
                            input="",
                            output=f"```sql\nSELECT {col.lower()}\nFROM {table} WITH (NOLOCK)\n```",
                            label=False  # Wrong case
                        ))

                        # Wrong case - uppercase
                        examples.append(KTOExample(
                            instruction=f"Query {col} column",
                            input="",
                            output=f"```sql\nSELECT {col.upper()}\nFROM {table} WITH (NOLOCK)\n```",
                            label=False  # Wrong case
                        ))

        return examples

    def generate_negative_aliases(self) -> List[KTOExample]:
        """Generate negative examples using table aliases."""
        examples = []

        alias_examples = [
            ("APTH", "a", "InvNum, GrossAmt"),
            ("JCJM", "j", "Job, Description"),
            ("PREH", "e", "Employee, FirstName"),
            ("GLDT", "g", "GLAcct, Amount"),
            ("ARCM", "c", "Customer, Name"),
            ("APVM", "v", "Vendor, Name"),
            ("EMEM", "eq", "Equipment, Description"),
            ("POHD", "po", "PO, Description"),
        ]

        for table, alias, cols in alias_examples:
            examples.append(KTOExample(
                instruction=f"Query {table} data",
                input="",
                output=f"```sql\nSELECT {alias}.{cols.split(',')[0]}\nFROM {table} {alias}\n```",
                label=False  # Using aliases
            ))

        return examples

    def generate_negative_missing_company(self) -> List[KTOExample]:
        """Generate negative examples missing company filter."""
        examples = []

        tables_with_company = [
            ("APTH", "APCo"),
            ("JCJM", "JCCo"),
            ("PRTH", "PRCo"),
            ("GLDT", "GLCo"),
            ("ARCM", "CustGroup"),
            ("EMEM", "EMCo"),
            ("POHD", "POCo"),
            ("SLHD", "SLCo"),
        ]

        for table, _ in tables_with_company:
            examples.append(KTOExample(
                instruction=f"Get all {table} records",
                input="",
                output=f"```sql\nSELECT *\nFROM {table} WITH (NOLOCK)\n```",
                label=False  # Missing company filter
            ))

        return examples

    def save_dataset(self, examples: List[KTOExample], output_path: str):
        """Save examples to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = [e.to_dict() for e in examples]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(examples)} KTO examples to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate KTO binary feedback data")
    parser.add_argument('--vgpt2', type=str, default='C:/Github/VGPT2',
                        help='Path to VGPT2 repository')
    parser.add_argument('--output', type=str, default='data/vgpt2_v3_kto.json',
                        help='Output file path')

    args = parser.parse_args()

    generator = KTOGenerator(args.vgpt2)
    examples = generator.generate_all()
    generator.save_dataset(examples, args.output)

    print(f"\nGenerated {len(examples)} KTO examples to {args.output}")


if __name__ == "__main__":
    main()
