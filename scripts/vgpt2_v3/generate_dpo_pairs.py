#!/usr/bin/env python3
"""
VGPT2 v3 DPO Preference Pairs Generator
=========================================
Generates preference pairs for Direct Preference Optimization (DPO) training.

DPO teaches the model to prefer correct SQL over incorrect SQL by showing
chosen (good) and rejected (bad) response pairs for the same instruction.

Target: 5,000+ preference pairs

Usage:
    python scripts/vgpt2_v3/generate_dpo_pairs.py --output data/vgpt2_v3_dpo.json
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DPOPair:
    """A DPO preference pair."""
    instruction: str
    input: str
    chosen: str
    rejected: str

    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "chosen": self.chosen,
            "rejected": self.rejected
        }


class DPOGenerator:
    """
    Generates DPO preference pairs for VGPT2.

    Categories:
    1. SQL correctness (WITH NOLOCK, company filter, etc.)
    2. Table naming (views vs base tables)
    3. Column naming (case sensitivity)
    4. JOIN patterns (complete vs incomplete)
    5. Hallucination (real vs fake tables)
    """

    def __init__(self, vgpt2_path: str):
        self.vgpt2 = Path(vgpt2_path)
        self.columns_data = {}
        self.fk_data = {}
        self.tables = []
        self._load_schema()

    def _load_schema(self):
        """Load schema data from VGPT2 repository."""
        # Load columns.json - try multiple possible locations
        columns_paths = [
            self.vgpt2 / "Viewpoint_Database" / "_MetadataV2" / "_data" / "columns.json",
            self.vgpt2 / "Viewpoint_Database" / "_Metadata" / "columns.json",
            self.vgpt2 / "structured_docs" / "schema" / "columns.json",
        ]

        for columns_file in columns_paths:
            if columns_file.exists():
                with open(columns_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                # Handle list format (each item has ObjectName and ColumnName)
                if isinstance(raw_data, list):
                    # Group by table name
                    from collections import defaultdict
                    table_columns = defaultdict(list)
                    for item in raw_data:
                        table_name = item.get('ObjectName', '')
                        if table_name:
                            table_columns[table_name].append({
                                'column_name': item.get('ColumnName', ''),
                                'data_type': item.get('DataType', ''),
                                'is_nullable': item.get('IsNullable', 'True'),
                            })
                    self.columns_data = dict(table_columns)
                else:
                    self.columns_data = raw_data

                self.tables = list(self.columns_data.keys())
                logger.info(f"Loaded {len(self.tables)} tables from {columns_file}")
                break

        # Load foreign_keys.json - try multiple possible locations
        fk_paths = [
            self.vgpt2 / "Viewpoint_Database" / "_Metadata" / "foreign_keys.json",
            self.vgpt2 / "structured_docs" / "schema" / "foreign_keys.json",
        ]

        for fk_file in fk_paths:
            if fk_file.exists():
                with open(fk_file, 'r', encoding='utf-8') as f:
                    raw_fk = json.load(f)

                # Handle list format
                if isinstance(raw_fk, list):
                    from collections import defaultdict
                    fk_by_table = defaultdict(list)
                    for item in raw_fk:
                        parent_table = item.get('ParentTable', '')
                        if parent_table:
                            fk_by_table[parent_table].append({
                                'referenced_table': item.get('ReferencedTable', ''),
                                'fk_columns': [item.get('ParentColumn', '')],
                                'referenced_columns': [item.get('ReferencedColumn', '')],
                            })
                    self.fk_data = dict(fk_by_table)
                else:
                    self.fk_data = raw_fk

                logger.info(f"Loaded FK relationships for {len(self.fk_data)} tables from {fk_file}")
                break

    def generate_all(self) -> List[DPOPair]:
        """Generate all DPO pairs."""
        pairs = []

        # Category 1: WITH (NOLOCK) pairs
        logger.info("Generating NOLOCK preference pairs...")
        pairs.extend(self.generate_nolock_pairs())

        # Category 2: View vs base table pairs
        logger.info("Generating view vs base table pairs...")
        pairs.extend(self.generate_view_vs_base_pairs())

        # Category 3: Company filter pairs
        logger.info("Generating company filter pairs...")
        pairs.extend(self.generate_company_filter_pairs())

        # Category 4: Complete vs incomplete JOIN pairs
        logger.info("Generating JOIN completeness pairs...")
        pairs.extend(self.generate_join_pairs())

        # Category 5: Alias vs full name pairs
        logger.info("Generating alias preference pairs...")
        pairs.extend(self.generate_alias_pairs())

        # Category 6: Column case pairs
        logger.info("Generating column case pairs...")
        pairs.extend(self.generate_case_pairs())

        # Category 7: Hallucination pairs
        logger.info("Generating hallucination preference pairs...")
        pairs.extend(self.generate_hallucination_pairs())

        # Category 8: Reporting view pairs
        logger.info("Generating reporting view pairs...")
        pairs.extend(self.generate_reporting_view_pairs())

        # Category 9: Month format pairs
        logger.info("Generating month format pairs...")
        pairs.extend(self.generate_month_format_pairs())

        # Category 10: SQL generation quality pairs
        logger.info("Generating SQL quality pairs...")
        pairs.extend(self.generate_sql_quality_pairs())

        # Category 11: Schema-based NOLOCK pairs (dynamic from columns.json)
        logger.info("Generating schema-based NOLOCK pairs...")
        pairs.extend(self.generate_schema_nolock_pairs())

        # Category 12: Schema-based case pairs (dynamic from columns.json)
        logger.info("Generating schema-based case pairs...")
        pairs.extend(self.generate_schema_case_pairs())

        # Category 13: FK-based JOIN pairs (dynamic from foreign_keys.json)
        logger.info("Generating FK-based JOIN pairs...")
        pairs.extend(self.generate_fk_join_pairs())

        # Category 14: More hallucination pairs
        logger.info("Generating extended hallucination pairs...")
        pairs.extend(self.generate_extended_hallucination_pairs())

        logger.info(f"Generated {len(pairs)} total DPO pairs")
        return pairs

    def generate_nolock_pairs(self) -> List[DPOPair]:
        """Generate pairs showing preference for WITH (NOLOCK)."""
        pairs = []

        tables = ["APTH", "APTL", "JCJM", "JCCD", "PRTH", "PREH", "GLDT", "ARCM", "ARTH"]

        for table in tables:
            pairs.append(DPOPair(
                instruction=f"Write SQL to select all from {table}",
                input="",
                chosen=f"```sql\nSELECT *\nFROM {table} WITH (NOLOCK)\n```",
                rejected=f"```sql\nSELECT *\nFROM {table}\n```"
            ))

            pairs.append(DPOPair(
                instruction=f"Query {table} for company 1",
                input="",
                chosen=f"```sql\nSELECT *\nFROM {table} WITH (NOLOCK)\nWHERE {self._get_company_col(table)} = 1\n```",
                rejected=f"```sql\nSELECT *\nFROM {table}\nWHERE {self._get_company_col(table)} = 1\n```"
            ))

        return pairs

    def generate_view_vs_base_pairs(self) -> List[DPOPair]:
        """Generate pairs showing preference for views over base tables."""
        pairs = []

        tables = [
            ("bAPTH", "APTH", "APCo"),
            ("bAPTL", "APTL", "APCo"),
            ("bJCJM", "JCJM", "JCCo"),
            ("bJCCD", "JCCD", "JCCo"),
            ("bPRTH", "PRTH", "PRCo"),
            ("bGLDT", "GLDT", "GLCo"),
            ("bARTH", "ARTH", "ARCo"),
        ]

        for base, view, co in tables:
            pairs.append(DPOPair(
                instruction=f"Select data from {view.lower()} table",
                input="",
                chosen=f"```sql\nSELECT *\nFROM {view} WITH (NOLOCK)\nWHERE {co} = @{co}\n```\n\nUsing the view {view} for SELECT queries ensures proper security and performance.",
                rejected=f"```sql\nSELECT *\nFROM {base}\nWHERE {co} = @{co}\n```"
            ))

            pairs.append(DPOPair(
                instruction=f"Query the {base} table",
                input="",
                chosen=f"For SELECT queries, use the view {view} instead of base table {base}:\n\n```sql\nSELECT *\nFROM {view} WITH (NOLOCK)\nWHERE {co} = @{co}\n```\n\nBase tables (b-prefix) should only be used for INSERT/UPDATE/DELETE operations.",
                rejected=f"```sql\nSELECT *\nFROM {base}\nWHERE {co} = @{co}\n```"
            ))

        return pairs

    def generate_company_filter_pairs(self) -> List[DPOPair]:
        """Generate pairs showing preference for company filtering."""
        pairs = []

        queries = [
            ("APTH", "APCo", "Get all AP invoices"),
            ("JCJM", "JCCo", "List all jobs"),
            ("PRTH", "PRCo", "Query payroll timecards"),
            ("GLDT", "GLCo", "Get GL transactions"),
            ("ARCM", "CustGroup", "List customers"),
        ]

        for table, col, desc in queries:
            pairs.append(DPOPair(
                instruction=desc,
                input="",
                chosen=f"```sql\nSELECT *\nFROM {table} WITH (NOLOCK)\nWHERE {col} = @{col}\n```\n\nAlways filter by {col} to ensure proper data isolation between companies.",
                rejected=f"```sql\nSELECT *\nFROM {table} WITH (NOLOCK)\n```"
            ))

        return pairs

    def generate_join_pairs(self) -> List[DPOPair]:
        """Generate pairs for complete vs incomplete JOINs."""
        pairs = []

        joins = [
            {
                "instruction": "Join APTH and APTL",
                "complete": """```sql
SELECT APTH.*, APTL.*
FROM APTH WITH (NOLOCK)
INNER JOIN APTL WITH (NOLOCK)
  ON APTH.APCo = APTL.APCo
  AND APTH.Mth = APTL.Mth
  AND APTH.APTrans = APTL.APTrans
WHERE APTH.APCo = @APCo
```""",
                "incomplete": """```sql
SELECT APTH.*, APTL.*
FROM APTH
JOIN APTL ON APTH.APTrans = APTL.APTrans
```"""
            },
            {
                "instruction": "Join JCJM with JCCD",
                "complete": """```sql
SELECT JCJM.Job, JCJM.Description, JCCD.ActualCost
FROM JCJM WITH (NOLOCK)
INNER JOIN JCCD WITH (NOLOCK)
  ON JCJM.JCCo = JCCD.JCCo
  AND JCJM.Job = JCCD.Job
WHERE JCJM.JCCo = @JCCo
```""",
                "incomplete": """```sql
SELECT JCJM.Job, JCJM.Description, JCCD.ActualCost
FROM JCJM
JOIN JCCD ON JCJM.Job = JCCD.Job
```"""
            },
            {
                "instruction": "Join APTH with APVM to get vendor names",
                "complete": """```sql
SELECT APTH.InvNum, APTH.GrossAmt, APVM.Name
FROM APTH WITH (NOLOCK)
INNER JOIN APVM WITH (NOLOCK)
  ON APTH.VendorGroup = APVM.VendorGroup
  AND APTH.Vendor = APVM.Vendor
WHERE APTH.APCo = @APCo
```""",
                "incomplete": """```sql
SELECT APTH.InvNum, APTH.GrossAmt, APVM.Name
FROM APTH
JOIN APVM ON APTH.Vendor = APVM.Vendor
```"""
            },
        ]

        for join in joins:
            pairs.append(DPOPair(
                instruction=join["instruction"],
                input="",
                chosen=join["complete"],
                rejected=join["incomplete"]
            ))

        return pairs

    def generate_alias_pairs(self) -> List[DPOPair]:
        """Generate pairs showing preference against table aliases."""
        pairs = []

        examples = [
            {
                "instruction": "Query APTH with vendor info",
                "with_alias": "SELECT a.InvNum, v.Name FROM APTH a JOIN APVM v ON a.Vendor = v.Vendor",
                "without_alias": """```sql
SELECT APTH.InvNum, APVM.Name
FROM APTH WITH (NOLOCK)
INNER JOIN APVM WITH (NOLOCK)
  ON APTH.VendorGroup = APVM.VendorGroup
  AND APTH.Vendor = APVM.Vendor
WHERE APTH.APCo = @APCo
```

Using full table names instead of aliases makes the query self-documenting and easier to validate."""
            },
            {
                "instruction": "Join job master with cost detail",
                "with_alias": "SELECT j.Job, c.ActualCost FROM JCJM j JOIN JCCD c ON j.Job = c.Job",
                "without_alias": """```sql
SELECT JCJM.Job, JCCD.ActualCost
FROM JCJM WITH (NOLOCK)
INNER JOIN JCCD WITH (NOLOCK)
  ON JCJM.JCCo = JCCD.JCCo
  AND JCJM.Job = JCCD.Job
WHERE JCJM.JCCo = @JCCo
```"""
            },
        ]

        for ex in examples:
            pairs.append(DPOPair(
                instruction=ex["instruction"],
                input="",
                chosen=ex["without_alias"],
                rejected=ex["with_alias"]
            ))

        return pairs

    def generate_case_pairs(self) -> List[DPOPair]:
        """Generate pairs for correct column case."""
        pairs = []

        case_examples = [
            {
                "instruction": "Get AP company and vendor from APTH",
                "correct": "SELECT APCo, Vendor FROM APTH WITH (NOLOCK)",
                "wrong": "SELECT apco, vendor FROM APTH WITH (NOLOCK)"
            },
            {
                "instruction": "Query JC company and job from JCJM",
                "correct": "SELECT JCCo, Job, Description FROM JCJM WITH (NOLOCK)",
                "wrong": "SELECT jcco, job, description FROM JCJM WITH (NOLOCK)"
            },
            {
                "instruction": "Get invoice number and amount",
                "correct": "SELECT InvNum, GrossAmt FROM APTH WITH (NOLOCK) WHERE APCo = 1",
                "wrong": "SELECT INVNUM, GROSSAMT FROM APTH WITH (NOLOCK) WHERE APCO = 1"
            },
        ]

        for ex in case_examples:
            pairs.append(DPOPair(
                instruction=ex["instruction"],
                input="",
                chosen=f"```sql\n{ex['correct']}\n```\n\nViewpoint uses exact column case (Latin1_General_BIN collation).",
                rejected=f"```sql\n{ex['wrong']}\n```"
            ))

        return pairs

    def generate_hallucination_pairs(self) -> List[DPOPair]:
        """Generate pairs for real vs hallucinated tables."""
        pairs = []

        fake_tables = [
            ("Invoice", "APTH", "AP Transaction Header"),
            ("Customer", "ARCM", "AR Customer Master"),
            ("Vendor", "APVM", "AP Vendor Master"),
            ("Employee", "PREH", "PR Employee Header"),
            ("Job", "JCJM", "JC Job Master"),
            ("Project", "JCJM", "JC Job Master"),
        ]

        for fake, real, description in fake_tables:
            pairs.append(DPOPair(
                instruction=f"Query the {fake} table",
                input="",
                chosen=f"There is no '{fake}' table in Viewpoint Vista. The correct table is {real} ({description}).\n\n```sql\nSELECT *\nFROM {real} WITH (NOLOCK)\nWHERE {self._get_company_col(real)} = @{self._get_company_col(real)}\n```",
                rejected=f"```sql\nSELECT *\nFROM {fake}\n```"
            ))

            pairs.append(DPOPair(
                instruction=f"What columns are in {fake}?",
                input="",
                chosen=f"'{fake}' is not a valid Viewpoint table. You're likely looking for {real} ({description}). Use the schema documentation to find the correct columns for {real}.",
                rejected=f"The {fake} table contains columns like {fake}ID, Name, Description, Status, CreateDate..."
            ))

        return pairs

    def generate_reporting_view_pairs(self) -> List[DPOPair]:
        """Generate pairs showing preference for reporting views."""
        pairs = []

        report_examples = [
            {
                "instruction": "Get AP aging report data",
                "custom": """SELECT APTH.Vendor, APTH.InvNum, APTH.InvDate, APTH.GrossAmt
FROM APTH WITH (NOLOCK)
JOIN APVM WITH (NOLOCK) ON APTH.VendorGroup = APVM.VendorGroup AND APTH.Vendor = APVM.Vendor
WHERE APTH.APCo = @APCo AND APTH.Status = 0""",
                "view": """Use the vrvAPAgingDetail reporting view which is pre-built for this purpose:

```sql
SELECT *
FROM vrvAPAgingDetail WITH (NOLOCK)
WHERE APCo = @APCo
```

Always check for vrv*/brv* reporting views before writing custom JOINs."""
            },
            {
                "instruction": "Query job cost and revenue",
                "custom": """SELECT JCCD.Job, SUM(JCCD.ActualCost), SUM(JCCD.ActualRevenue)
FROM JCCD WITH (NOLOCK)
GROUP BY JCCD.Job""",
                "view": """Use the brvJCCostRevenue reporting view:

```sql
SELECT *
FROM brvJCCostRevenue WITH (NOLOCK)
WHERE JCCo = @JCCo
```

Reporting views (vrv*/brv*) are optimized and validated for Crystal Reports."""
            },
        ]

        for ex in report_examples:
            pairs.append(DPOPair(
                instruction=ex["instruction"],
                input="",
                chosen=ex["view"],
                rejected=ex["custom"]
            ))

        return pairs

    def generate_month_format_pairs(self) -> List[DPOPair]:
        """Generate pairs for correct month format."""
        pairs = []

        month_examples = [
            ("2024-01", "2024-01-01"),
            ("Jan 2024", "2024-01-01"),
            ("1/2024", "2024-01-01"),
            ("202401", "2024-01-01"),
        ]

        for wrong, correct in month_examples:
            pairs.append(DPOPair(
                instruction=f"Filter APTH for month {wrong}",
                input="",
                chosen=f"```sql\nSELECT *\nFROM APTH WITH (NOLOCK)\nWHERE APCo = @APCo\n  AND Mth = '{correct}'\n```\n\nViewpoint stores Mth columns as the first day of the month (e.g., '{correct}').",
                rejected=f"```sql\nSELECT *\nFROM APTH WITH (NOLOCK)\nWHERE Mth = '{wrong}'\n```"
            ))

        return pairs

    def generate_sql_quality_pairs(self) -> List[DPOPair]:
        """Generate pairs for overall SQL quality."""
        pairs = []

        quality_examples = [
            {
                "instruction": "Find unpaid AP invoices over $10,000",
                "good": """```sql
SELECT
  APTH.APCo,
  APTH.Vendor,
  APVM.Name AS VendorName,
  APTH.InvNum,
  APTH.InvDate,
  APTH.GrossAmt
FROM APTH WITH (NOLOCK)
INNER JOIN APVM WITH (NOLOCK)
  ON APTH.VendorGroup = APVM.VendorGroup
  AND APTH.Vendor = APVM.Vendor
WHERE APTH.APCo = @APCo
  AND APTH.Status = 0  -- Unpaid
  AND APTH.GrossAmt > 10000
ORDER BY APTH.GrossAmt DESC
```

This query:
- Uses views with WITH (NOLOCK)
- Filters by company
- Includes vendor name via proper JOIN
- Orders by amount descending""",
                "bad": "SELECT * FROM Invoice WHERE Amount > 10000 AND Paid = false"
            },
            {
                "instruction": "Get all active jobs with their total cost",
                "good": """```sql
SELECT
  JCJM.JCCo,
  JCJM.Job,
  JCJM.Description,
  SUM(JCCD.ActualCost) AS TotalCost
FROM JCJM WITH (NOLOCK)
INNER JOIN JCCD WITH (NOLOCK)
  ON JCJM.JCCo = JCCD.JCCo
  AND JCJM.Job = JCCD.Job
WHERE JCJM.JCCo = @JCCo
  AND JCJM.JobStatus = 1  -- Active
GROUP BY JCJM.JCCo, JCJM.Job, JCJM.Description
ORDER BY TotalCost DESC
```""",
                "bad": "SELECT Job, Cost FROM Jobs WHERE Active = 1"
            },
        ]

        for ex in quality_examples:
            pairs.append(DPOPair(
                instruction=ex["instruction"],
                input="",
                chosen=ex["good"],
                rejected=ex["bad"]
            ))

        return pairs

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

    def generate_schema_nolock_pairs(self) -> List[DPOPair]:
        """Generate NOLOCK pairs from actual schema tables."""
        import random
        pairs = []

        # Get a sample of common tables
        common_tables = [t for t in self.tables if not t.startswith('b') and not t.startswith('vrv') and not t.startswith('brv')]
        sample_tables = random.sample(common_tables, min(200, len(common_tables)))

        for table in sample_tables:
            if table in self.columns_data:
                cols = self.columns_data[table]
                if cols:
                    # Get first 3 column names
                    col_names = [c.get('column_name', '') for c in cols[:3] if c.get('column_name')]
                    if col_names:
                        select_cols = ", ".join(col_names)
                        co_col = self._get_company_col(table)

                        pairs.append(DPOPair(
                            instruction=f"Select {', '.join(col_names[:2])} from {table}",
                            input="",
                            chosen=f"```sql\nSELECT {select_cols}\nFROM {table} WITH (NOLOCK)\nWHERE {co_col} = @{co_col}\n```",
                            rejected=f"```sql\nSELECT {select_cols}\nFROM {table}\n```"
                        ))

        return pairs

    def generate_schema_case_pairs(self) -> List[DPOPair]:
        """Generate case sensitivity pairs from actual column data."""
        import random
        pairs = []

        sample_tables = random.sample(self.tables, min(300, len(self.tables)))

        for table in sample_tables:
            if table in self.columns_data:
                cols = self.columns_data[table]
                if cols and len(cols) >= 2:
                    col1 = cols[0].get('column_name', '')
                    col2 = cols[1].get('column_name', '') if len(cols) > 1 else ''

                    if col1 and col2:
                        co_col = self._get_company_col(table)

                        pairs.append(DPOPair(
                            instruction=f"Query {col1} and {col2} from {table}",
                            input="",
                            chosen=f"```sql\nSELECT {col1}, {col2}\nFROM {table} WITH (NOLOCK)\nWHERE {co_col} = @{co_col}\n```\n\nViewpoint uses Latin1_General_BIN collation - column names are case-sensitive.",
                            rejected=f"```sql\nSELECT {col1.lower()}, {col2.lower()}\nFROM {table} WITH (NOLOCK)\n```"
                        ))

        return pairs

    def generate_fk_join_pairs(self) -> List[DPOPair]:
        """Generate JOIN pairs from foreign key relationships."""
        import random
        pairs = []

        # Sample FK relationships
        fk_items = list(self.fk_data.items()) if self.fk_data else []
        sample_fks = random.sample(fk_items, min(500, len(fk_items)))

        for table, relations in sample_fks:
            if not relations:
                continue

            for rel in relations[:2]:  # Max 2 relations per table
                ref_table = rel.get('referenced_table', '')
                fk_cols = rel.get('fk_columns', [])
                ref_cols = rel.get('referenced_columns', [])

                if ref_table and fk_cols and ref_cols and len(fk_cols) == len(ref_cols):
                    co_col = self._get_company_col(table)

                    # Build complete JOIN
                    join_conditions = " AND ".join([f"{table}.{fk} = {ref_table}.{ref}" for fk, ref in zip(fk_cols, ref_cols)])

                    # Get some columns
                    t1_cols = [c.get('column_name', '') for c in self.columns_data.get(table, [])[:2] if c.get('column_name')]
                    t2_cols = [c.get('column_name', '') for c in self.columns_data.get(ref_table, [])[:2] if c.get('column_name')]

                    if t1_cols and t2_cols:
                        select_clause = ", ".join([f"{table}.{c}" for c in t1_cols] + [f"{ref_table}.{c}" for c in t2_cols])

                        pairs.append(DPOPair(
                            instruction=f"Join {table} with {ref_table}",
                            input="",
                            chosen=f"""```sql
SELECT {select_clause}
FROM {table} WITH (NOLOCK)
INNER JOIN {ref_table} WITH (NOLOCK)
  ON {join_conditions}
WHERE {table}.{co_col} = @{co_col}
```

Using the complete foreign key relationship ensures data integrity.""",
                            rejected=f"""```sql
SELECT *
FROM {table}
JOIN {ref_table} ON {table}.{fk_cols[0]} = {ref_table}.{ref_cols[0]}
```"""
                        ))

        return pairs

    def generate_extended_hallucination_pairs(self) -> List[DPOPair]:
        """Generate more hallucination rejection pairs."""
        pairs = []

        # Common fake table names that don't exist
        fake_tables = [
            ("Invoices", "APTH", "AP Transaction Header"),
            ("APInvoice", "APTH", "AP Transaction Header"),
            ("VendorInvoice", "APTH", "AP Transaction Header"),
            ("Customers", "ARCM", "AR Customer Master"),
            ("CustomerMaster", "ARCM", "AR Customer Master"),
            ("Vendors", "APVM", "AP Vendor Master"),
            ("VendorMaster", "APVM", "AP Vendor Master"),
            ("Employees", "PREH", "PR Employee Header"),
            ("EmployeeMaster", "PREH", "PR Employee Header"),
            ("Jobs", "JCJM", "JC Job Master"),
            ("JobMaster", "JCJM", "JC Job Master"),
            ("Projects", "JCJM", "JC Job Master"),
            ("ProjectMaster", "JCJM", "JC Job Master"),
            ("WorkOrders", "JCJM", "JC Job Master"),
            ("Transactions", "GLDT", "GL Detail"),
            ("GLTransactions", "GLDT", "GL Detail"),
            ("Accounts", "GLAC", "GL Account"),
            ("ChartOfAccounts", "GLAC", "GL Account"),
            ("PurchaseOrders", "POHD", "PO Header"),
            ("POHeader", "POHD", "PO Header"),
            ("Timecards", "PRTH", "PR Timecard Header"),
            ("TimeCards", "PRTH", "PR Timecard Header"),
            ("PayrollTimecard", "PRTH", "PR Timecard Header"),
            ("Equipment", "EMEM", "EM Equipment Master"),
            ("EquipmentMaster", "EMEM", "EM Equipment Master"),
            ("Materials", "INMT", "IN Material"),
            ("Inventory", "INMT", "IN Material"),
            ("Subcontracts", "SLHD", "SL Header"),
            ("SubcontractHeader", "SLHD", "SL Header"),
            ("Contracts", "JCCM", "JC Contract Master"),
            ("ContractMaster", "JCCM", "JC Contract Master"),
            ("CostCodes", "JCCH", "JC Cost/Phase Header"),
            ("Phases", "JCCH", "JC Cost/Phase Header"),
            ("ChangeOrders", "JCOI", "JC Change Order Item"),
            ("Billing", "ARBH", "AR Bill Header"),
            ("ARBilling", "ARBH", "AR Bill Header"),
        ]

        for fake, real, description in fake_tables:
            co_col = self._get_company_col(real)

            pairs.append(DPOPair(
                instruction=f"Query the {fake} table",
                input="",
                chosen=f"There is no '{fake}' table in Viewpoint Vista. The correct table is {real} ({description}).\n\n```sql\nSELECT *\nFROM {real} WITH (NOLOCK)\nWHERE {co_col} = @{co_col}\n```",
                rejected=f"```sql\nSELECT *\nFROM {fake}\n```"
            ))

            pairs.append(DPOPair(
                instruction=f"Select all from {fake}",
                input="",
                chosen=f"'{fake}' does not exist. Use {real} ({description}) instead:\n\n```sql\nSELECT *\nFROM {real} WITH (NOLOCK)\nWHERE {co_col} = @{co_col}\n```",
                rejected=f"```sql\nSELECT *\nFROM {fake} WITH (NOLOCK)\n```"
            ))

        return pairs

    def save_dataset(self, pairs: List[DPOPair], output_path: str):
        """Save pairs to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = [p.to_dict() for p in pairs]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(pairs)} DPO pairs to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate DPO preference pairs")
    parser.add_argument('--vgpt2', type=str, default='C:/Github/VGPT2',
                        help='Path to VGPT2 repository')
    parser.add_argument('--output', type=str, default='data/vgpt2_v3_dpo.json',
                        help='Output file path')

    args = parser.parse_args()

    generator = DPOGenerator(args.vgpt2)
    pairs = generator.generate_all()
    generator.save_dataset(pairs, args.output)

    print(f"\nGenerated {len(pairs)} DPO pairs to {args.output}")


if __name__ == "__main__":
    main()
