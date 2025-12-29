#!/usr/bin/env python3
"""
VGPT2 v3 Negative Example Generator
====================================
Generates negative training examples to prevent hallucination.

These examples teach the model to:
- Say "doesn't exist" for non-existent tables
- Reject incorrect SQL patterns
- Correct common mistakes
- Refuse to generate invalid queries

Target: 2,000+ negative examples

Usage:
    python scripts/vgpt2_v3/generate_negative_examples.py --output data/vgpt2_v3_negative.json
"""

import json
import logging
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class NegativeExample:
    """A negative training example."""
    instruction: str
    input: str = ""
    output: str = ""
    category: str = "negative_example"

    def to_alpaca(self) -> Dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }


class NegativeExampleGenerator:
    """
    Generates negative examples for VGPT2 training.

    Categories:
    1. Non-existent tables/views
    2. Non-existent columns
    3. Invalid SQL patterns
    4. Wrong JOIN patterns
    5. Case sensitivity errors
    """

    # Common fake table names users might ask about (expanded for comprehensive coverage)
    FAKE_TABLES = [
        # Invoice variations
        "Invoice", "Invoices", "InvoiceHeader", "InvoiceDetail", "InvoiceLine",
        "APInvoice", "APInvoiceHeader", "VendorInvoice", "InvoiceHistory",
        "ARInvoice", "CustomerInvoice", "BillingInvoice",
        # Customer variations
        "Customer", "Customers", "CustomerMaster", "CustomerInfo", "CustomerData",
        "CustomerAccount", "CustomerProfile", "Client", "Clients",
        # Vendor variations
        "Vendor", "Vendors", "VendorMaster", "VendorInfo", "VendorData",
        "Supplier", "Suppliers", "SupplierMaster",
        # Employee variations
        "Employee", "Employees", "EmployeeMaster", "EmployeeInfo", "EmployeeData",
        "Staff", "Personnel", "Worker", "Workers", "TeamMember",
        # Order variations
        "Order", "Orders", "OrderHeader", "OrderDetail", "OrderLine",
        "SalesOrder", "PurchaseOrder", "WorkOrder", "ServiceOrder",
        # Payment variations
        "Payment", "Payments", "PaymentHistory", "PaymentDetail",
        "PaymentTransaction", "PaymentRecord", "VendorPayment",
        # Product variations
        "Product", "Products", "ProductMaster", "Item", "Items", "ItemMaster",
        # Sales variations
        "Sales", "SalesOrder", "SalesDetail", "SalesData", "SalesHistory",
        # Purchase variations
        "Purchase", "Purchases", "PurchaseOrder", "PurchaseDetail", "Purchasing",
        # Transaction variations
        "Transaction", "Transactions", "TransactionHistory", "TransactionDetail",
        "TransactionLog", "GLTransaction", "APTransaction", "ARTransaction",
        # Account variations
        "Account", "Accounts", "AccountMaster", "ChartOfAccounts", "GLAccount",
        "AccountInfo", "BankAccount", "AccountBalance",
        # Project/Job variations
        "Project", "Projects", "ProjectMaster", "ProjectDetail", "ProjectInfo",
        "Job", "Jobs", "JobMaster", "JobDetail", "JobInfo", "JobCost",
        # Contract variations
        "Contract", "Contracts", "ContractMaster", "ContractDetail", "Agreement",
        # Budget variations
        "Budget", "Budgets", "BudgetDetail", "BudgetLine", "BudgetHistory",
        # Cost variations
        "Cost", "Costs", "CostCenter", "CostDetail", "CostHistory", "CostType",
        # Department variations
        "Department", "Departments", "DepartmentMaster", "Division", "Divisions",
        # User variations
        "User", "Users", "UserProfile", "UserAccount", "UserInfo",
        # Report variations
        "Report", "Reports", "ReportData", "ReportHistory", "ReportDetail",
        # Inventory variations
        "Inventory", "InventoryItem", "Stock", "StockItem", "Warehouse",
        # Payroll variations
        "Payroll", "PayrollHeader", "PayrollDetail", "PayrollHistory",
        "PayCheck", "PayStub", "Wage", "Wages",
        # Time variations
        "TimeSheet", "TimeEntry", "TimeCard", "TimeClock", "TimeRecord",
        "Timecard", "TimeDetail", "WorkHours", "LaborHours",
        # Equipment variations
        "Equipment", "EquipmentMaster", "Asset", "Assets", "AssetMaster",
        "FixedAsset", "Machine", "Machinery", "Tool", "Tools",
        # Material variations
        "Material", "Materials", "MaterialStock", "MaterialUsage", "InventoryMaterial",
        # Labor variations
        "Labor", "LaborCost", "LaborHours", "LaborDetail", "LaborRate",
        # Billing variations
        "Billing", "BillingHistory", "Bill", "Bills", "BillingDetail",
        # Receipt variations
        "Receipt", "Receipts", "CashReceipt", "ReceiptHistory", "PaymentReceipt",
        # Check variations
        "Check", "Checks", "CheckHistory", "CheckDetail", "BankCheck",
        # Journal variations
        "Journal", "JournalEntry", "JournalDetail", "JournalHeader", "GLJournal",
        # Ledger variations
        "Ledger", "LedgerEntry", "GeneralLedger", "SubLedger", "LedgerDetail",
        # Tax variations
        "Tax", "Taxes", "TaxHistory", "TaxDetail", "SalesTax", "TaxRate",
        # Company variations
        "Company", "Companies", "CompanyMaster", "CompanyInfo", "Organization",
        # Address variations
        "Address", "Addresses", "AddressBook", "AddressMaster", "Location",
        # Contact variations
        "Contact", "Contacts", "ContactInfo", "ContactPerson", "ContactList",
        # Note/Document variations
        "Note", "Notes", "NoteHistory", "Memo", "Memos",
        "Document", "Documents", "DocumentStore", "DocumentHistory",
        "Attachment", "Attachments", "FileStore", "File", "Files",
        # Phase/CostCode variations
        "Phase", "Phases", "PhaseMaster", "CostCode", "CostCodes",
        # Retainage variations
        "Retainage", "RetainageHistory", "RetainageDetail",
        # Subcontract variations
        "Subcontract", "Subcontracts", "SubcontractMaster", "SubcontractDetail",
        # Change Order variations
        "ChangeOrder", "ChangeOrders", "CO", "Amendment", "Amendments",
        # Commitment variations
        "Commitment", "Commitments", "CommittedCost",
        # Forecast variations
        "Forecast", "Forecasts", "ForecastDetail", "Projection", "Projections",
        # Revenue variations
        "Revenue", "RevenueDetail", "RevenueHistory", "Income",
        # WIP variations
        "WIP", "WorkInProgress", "WIPDetail",
        # Miscellaneous common mistakes
        "Master", "Header", "Detail", "Line", "History", "Log", "Record",
        "Table", "Data", "Info", "List", "Summary", "Total",
    ]

    # Actual Viewpoint tables for suggestions
    ACTUAL_TABLES = {
        "Invoice": ["APTH (AP Transaction Header)", "vrvAP_MVAllInvoices", "APTL (AP Transaction Lines)"],
        "Invoices": ["APTH (AP Transaction Header)", "vrvAP_MVAllInvoices", "APTL (AP Transaction Lines)"],
        "Customer": ["ARCM (AR Customer Master)", "ARTH (AR Transaction Header)"],
        "Customers": ["ARCM (AR Customer Master)", "ARTH (AR Transaction Header)"],
        "Vendor": ["APVM (AP Vendor Master)", "APTH (AP Transaction Header)"],
        "Vendors": ["APVM (AP Vendor Master)"],
        "Employee": ["PREH (PR Employee Header)", "PRTH (PR Timecard Header)"],
        "Employees": ["PREH (PR Employee Header)"],
        "Order": ["POHD (PO Header)", "PODL (PO Detail)"],
        "Orders": ["POHD (PO Header)", "PODL (PO Detail)"],
        "Payment": ["APCM (AP Check Master)", "ARCR (AR Cash Receipt)"],
        "Payments": ["APCM (AP Check Master)", "ARCR (AR Cash Receipt)"],
        "Project": ["JCJM (JC Job Master)", "JCCI (JC Contract Items)"],
        "Projects": ["JCJM (JC Job Master)"],
        "Contract": ["JCJM (JC Job Master)", "JCCI (JC Contract Items)"],
        "Budget": ["JCCD (JC Cost Detail)", "JCCP (JC Cost Projections)"],
        "Job": ["JCJM (JC Job Master)", "JCCD (JC Cost Detail)"],
        "Jobs": ["JCJM (JC Job Master)"],
        "Cost": ["JCCD (JC Cost Detail)", "JCCH (JC Cost History)"],
        "Account": ["GLAC (GL Account)", "GLDT (GL Detail)"],
        "Accounts": ["GLAC (GL Account)"],
        "Equipment": ["EMEM (EM Equipment Master)", "EMRD (EM Revenue Detail)"],
        "Payroll": ["PRTH (PR Timecard Header)", "PRTD (PR Timecard Detail)"],
        "TimeSheet": ["PRTH (PR Timecard Header)", "PRTD (PR Timecard Detail)"],
    }

    # Wrong column names and their corrections
    WRONG_COLUMNS = {
        ("APTH", "InvoiceNumber"): "InvNum",
        ("APTH", "InvoiceID"): "APTrans",
        ("APTH", "VendorID"): "Vendor",
        ("APTH", "Amount"): "GrossAmt",
        ("APTH", "Company"): "APCo",
        ("JCJM", "JobNumber"): "Job",
        ("JCJM", "JobName"): "Description",
        ("JCJM", "ProjectID"): "Job",
        ("JCCD", "CostAmount"): "ActualCost",
        ("JCCD", "BudgetAmount"): "OrigEstCost",
        ("PREH", "EmployeeID"): "Employee",
        ("PREH", "EmployeeName"): "LastName, FirstName",
        ("ARCM", "CustomerID"): "Customer",
        ("ARCM", "CustomerName"): "Name",
        ("GLAC", "AccountNumber"): "GLAcct",
        ("GLAC", "AccountName"): "Description",
    }

    # Invalid SQL patterns
    INVALID_PATTERNS = [
        {
            "wrong": "SELECT * FROM bAPTH WHERE APCo = 1",
            "problem": "Using base table instead of view",
            "correct": "SELECT * FROM APTH WITH (NOLOCK) WHERE APCo = 1"
        },
        {
            "wrong": "SELECT * FROM APTH WHERE APCo = 1",
            "problem": "Missing WITH (NOLOCK)",
            "correct": "SELECT * FROM APTH WITH (NOLOCK) WHERE APCo = 1"
        },
        {
            "wrong": "SELECT a.* FROM APTH a WHERE a.APCo = 1",
            "problem": "Using table alias",
            "correct": "SELECT APTH.* FROM APTH WITH (NOLOCK) WHERE APTH.APCo = 1"
        },
        {
            "wrong": "SELECT * FROM dbo.APTH WITH (NOLOCK)",
            "problem": "Unnecessary schema prefix",
            "correct": "SELECT * FROM APTH WITH (NOLOCK)"
        },
        {
            "wrong": "SELECT apco, vendor FROM APTH",
            "problem": "Wrong column case",
            "correct": "SELECT APCo, Vendor FROM APTH WITH (NOLOCK)"
        },
        {
            "wrong": "SELECT * FROM APTH WHERE Mth = '2024-01'",
            "problem": "Invalid month format",
            "correct": "SELECT * FROM APTH WITH (NOLOCK) WHERE Mth = '2024-01-01'"
        },
        {
            "wrong": "SELECT * FROM APTH JOIN APTL ON APTH.APTrans = APTL.APTrans",
            "problem": "Incomplete JOIN (missing company and month)",
            "correct": "SELECT * FROM APTH WITH (NOLOCK) JOIN APTL WITH (NOLOCK) ON APTH.APCo = APTL.APCo AND APTH.Mth = APTL.Mth AND APTH.APTrans = APTL.APTrans"
        },
        {
            "wrong": "SELECT * FROM APVM WHERE APCo = 1",
            "problem": "APVM doesn't have APCo column",
            "correct": "SELECT * FROM APVM WITH (NOLOCK) WHERE VendorGroup = @VendorGroup"
        },
    ]

    def __init__(self, vgpt2_path: str):
        self.vgpt2 = Path(vgpt2_path)
        self.metadata_dir = self.vgpt2 / "Viewpoint_Database" / "_Metadata"
        self._actual_tables: Set[str] = set()
        self._load_actual_tables()

    def _load_actual_tables(self):
        """Load actual table names from metadata."""
        tables_file = self.metadata_dir / "_Viewpoint_ALL_Views_Tables_Complete.json"
        if tables_file.exists():
            with open(tables_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                name = item.get('name', item.get('TABLE_NAME', ''))
                if name:
                    self._actual_tables.add(name)
                    self._actual_tables.add(name.upper())

        # Also load from columns.json
        columns_file = self.metadata_dir / "columns.json"
        if columns_file.exists():
            with open(columns_file, 'r', encoding='utf-8') as f:
                columns = json.load(f)
            for col in columns:
                if isinstance(col, dict):
                    obj_name = col.get('ObjectName', '')
                    if obj_name:
                        self._actual_tables.add(obj_name)

        logger.info(f"Loaded {len(self._actual_tables)} actual table names")

    def generate_all(self) -> List[NegativeExample]:
        """Generate all negative examples."""
        examples = []

        # Category 1: Non-existent tables (500+)
        logger.info("Generating non-existent table examples...")
        examples.extend(self.generate_fake_table_examples())

        # Category 2: Wrong column names (200+)
        logger.info("Generating wrong column examples...")
        examples.extend(self.generate_wrong_column_examples())

        # Category 3: Invalid SQL patterns (200+)
        logger.info("Generating invalid SQL pattern examples...")
        examples.extend(self.generate_invalid_pattern_examples())

        # Category 4: Wrong JOINs (200+)
        logger.info("Generating wrong JOIN examples...")
        examples.extend(self.generate_wrong_join_examples())

        # Category 5: Generic/vague questions (200+)
        logger.info("Generating generic question examples...")
        examples.extend(self.generate_generic_question_examples())

        # Category 6: Case sensitivity errors (200+)
        logger.info("Generating case sensitivity examples...")
        examples.extend(self.generate_case_sensitivity_examples())

        logger.info(f"Generated {len(examples)} total negative examples")
        return examples

    def generate_fake_table_examples(self) -> List[NegativeExample]:
        """Generate examples for non-existent tables."""
        examples = []

        for fake_table in self.FAKE_TABLES:
            # Skip if it's actually a real table
            if fake_table in self._actual_tables:
                continue

            # Get suggestions
            suggestions = self.ACTUAL_TABLES.get(fake_table, [])
            if not suggestions:
                # Try to infer suggestions
                suggestions = self._infer_suggestions(fake_table)

            suggestion_text = ""
            if suggestions:
                suggestion_text = f"\n\nYou may be looking for:\n" + "\n".join(f"- {s}" for s in suggestions[:3])

            # Type 1: "What columns are in X?"
            examples.append(NegativeExample(
                instruction=f"What columns are in the {fake_table} table?",
                output=f"There is no table or view named '{fake_table}' in Viewpoint Vista. The database uses module-prefixed names like APTH (AP Transaction Header) or JCJM (JC Job Master).{suggestion_text}"
            ))

            # Type 2: "Describe the X table"
            examples.append(NegativeExample(
                instruction=f"Describe the {fake_table} table structure",
                output=f"The table '{fake_table}' does not exist in Viewpoint Vista. Viewpoint uses specific naming conventions with module prefixes (AP, AR, JC, GL, PR, etc.).{suggestion_text}"
            ))

            # Type 3: "Write SQL to query X"
            examples.append(NegativeExample(
                instruction=f"Write SQL to query the {fake_table} table",
                output=f"I cannot write a query for '{fake_table}' because this table does not exist in Viewpoint Vista.{suggestion_text}\n\nPlease specify the correct Viewpoint table name."
            ))

            # Type 4: "How do I join X with Y?"
            if random.random() < 0.3:  # Only for some
                other_fake = random.choice(self.FAKE_TABLES)
                if other_fake != fake_table:
                    examples.append(NegativeExample(
                        instruction=f"How do I join {fake_table} with {other_fake}?",
                        output=f"Neither '{fake_table}' nor '{other_fake}' exist in Viewpoint Vista. Viewpoint tables use module prefixes. For example:\n- AP tables: APTH, APTL, APVM\n- JC tables: JCJM, JCCD, JCCI\n\nPlease specify the correct Viewpoint table names."
                    ))

        return examples

    def _infer_suggestions(self, fake_name: str) -> List[str]:
        """Infer table suggestions based on fake name."""
        suggestions = []
        fake_lower = fake_name.lower()

        mappings = {
            "invoice": ["APTH (AP Transaction Header)", "APTL (AP Transaction Lines)"],
            "customer": ["ARCM (AR Customer Master)"],
            "vendor": ["APVM (AP Vendor Master)"],
            "employee": ["PREH (PR Employee Header)"],
            "job": ["JCJM (JC Job Master)"],
            "project": ["JCJM (JC Job Master)"],
            "payment": ["APCM (AP Check Master)"],
            "order": ["POHD (PO Header)"],
            "account": ["GLAC (GL Account)"],
            "cost": ["JCCD (JC Cost Detail)"],
            "budget": ["JCCD (JC Cost Detail)"],
            "equipment": ["EMEM (EM Equipment Master)"],
            "payroll": ["PRTH (PR Timecard Header)"],
            "time": ["PRTH (PR Timecard Header)"],
        }

        for keyword, tables in mappings.items():
            if keyword in fake_lower:
                suggestions.extend(tables)

        return list(set(suggestions))[:3]

    def generate_wrong_column_examples(self) -> List[NegativeExample]:
        """Generate examples for wrong column names."""
        examples = []

        for (table, wrong_col), correct_col in self.WRONG_COLUMNS.items():
            # Type 1: Query using wrong column
            examples.append(NegativeExample(
                instruction=f"Write SQL to get {wrong_col} from {table}",
                output=f"The column '{wrong_col}' does not exist in {table}. The correct column name is '{correct_col}'.\n\nCorrect query:\n```sql\nSELECT {correct_col}\nFROM {table} WITH (NOLOCK)\n```\n\nNote: Viewpoint uses specific column naming conventions. Always verify column names against the schema."
            ))

            # Type 2: Filter using wrong column
            examples.append(NegativeExample(
                instruction=f"Filter {table} where {wrong_col} = 123",
                output=f"The column '{wrong_col}' does not exist in {table}. You should use '{correct_col}' instead.\n\n```sql\nSELECT *\nFROM {table} WITH (NOLOCK)\nWHERE {correct_col} = 123\n```"
            ))

        return examples

    def generate_invalid_pattern_examples(self) -> List[NegativeExample]:
        """Generate examples for invalid SQL patterns."""
        examples = []

        for pattern in self.INVALID_PATTERNS:
            # Type 1: "Is this SQL correct?"
            examples.append(NegativeExample(
                instruction="Is this SQL query correct for Viewpoint Vista?",
                input=pattern["wrong"],
                output=f"No, this query has an issue: {pattern['problem']}.\n\nCorrected query:\n```sql\n{pattern['correct']}\n```"
            ))

            # Type 2: "Fix this SQL"
            examples.append(NegativeExample(
                instruction="Fix this Viewpoint SQL query",
                input=pattern["wrong"],
                output=f"The query has a problem: {pattern['problem']}.\n\nFixed version:\n```sql\n{pattern['correct']}\n```"
            ))

            # Type 3: "Run this query"
            examples.append(NegativeExample(
                instruction="Help me run this query against Viewpoint",
                input=pattern["wrong"],
                output=f"This query needs correction before running. Issue: {pattern['problem']}.\n\nUse this corrected version:\n```sql\n{pattern['correct']}\n```"
            ))

        return examples

    def generate_wrong_join_examples(self) -> List[NegativeExample]:
        """Generate examples for incorrect JOIN patterns."""
        examples = []

        wrong_joins = [
            {
                "tables": ("APTH", "APVM"),
                "wrong": "APTH.Vendor = APVM.Vendor",
                "problem": "Missing VendorGroup in JOIN",
                "correct": "APTH.VendorGroup = APVM.VendorGroup AND APTH.Vendor = APVM.Vendor"
            },
            {
                "tables": ("APTH", "APTL"),
                "wrong": "APTH.APTrans = APTL.APTrans",
                "problem": "Missing APCo and Mth in JOIN",
                "correct": "APTH.APCo = APTL.APCo AND APTH.Mth = APTL.Mth AND APTH.APTrans = APTL.APTrans"
            },
            {
                "tables": ("JCJM", "JCCD"),
                "wrong": "JCJM.Job = JCCD.Job",
                "problem": "Missing JCCo in JOIN",
                "correct": "JCJM.JCCo = JCCD.JCCo AND JCJM.Job = JCCD.Job"
            },
            {
                "tables": ("ARTH", "ARCM"),
                "wrong": "ARTH.Customer = ARCM.Customer",
                "problem": "Missing CustGroup in JOIN",
                "correct": "ARTH.CustGroup = ARCM.CustGroup AND ARTH.Customer = ARCM.Customer"
            },
            {
                "tables": ("PRTH", "PREH"),
                "wrong": "PRTH.Employee = PREH.Employee",
                "problem": "Missing PRCo in JOIN",
                "correct": "PRTH.PRCo = PREH.PRCo AND PRTH.Employee = PREH.Employee"
            },
        ]

        for join in wrong_joins:
            t1, t2 = join["tables"]

            examples.append(NegativeExample(
                instruction=f"How do I join {t1} with {t2}?",
                output=f"To correctly join {t1} with {t2}, you must include all key columns:\n\n```sql\nSELECT *\nFROM {t1} WITH (NOLOCK)\nINNER JOIN {t2} WITH (NOLOCK)\n  ON {join['correct']}\n```\n\nIncomplete JOIN like `{join['wrong']}` would be incorrect because: {join['problem']}."
            ))

            # Wrong example to fix
            wrong_sql = f"SELECT * FROM {t1} JOIN {t2} ON {join['wrong']}"
            examples.append(NegativeExample(
                instruction="Fix this JOIN query",
                input=wrong_sql,
                output=f"This JOIN is incomplete. {join['problem']}.\n\nCorrected:\n```sql\nSELECT *\nFROM {t1} WITH (NOLOCK)\nINNER JOIN {t2} WITH (NOLOCK)\n  ON {join['correct']}\n```"
            ))

        return examples

    def generate_generic_question_examples(self) -> List[NegativeExample]:
        """Generate examples for overly generic questions."""
        examples = []

        generic_questions = [
            {
                "q": "How do I query invoices?",
                "a": "I need more specific information to help you query invoices:\n\n1. **AP Invoices (Vendor):** Use `APTH` (header) and `APTL` (lines)\n2. **AR Invoices (Customer):** Use `ARTH` (header) and `ARTL` (lines)\n3. **Job Billing:** Use `JCCD` with appropriate type filters\n\nPlease specify which type of invoice and what information you need."
            },
            {
                "q": "Get all transactions",
                "a": "\"Transactions\" is too generic for Viewpoint Vista. Please specify:\n\n- **AP Transactions:** APTH/APTL\n- **AR Transactions:** ARTH/ARTL\n- **GL Transactions:** GLDT\n- **JC Transactions:** JCCD\n- **PR Transactions:** PRTH/PRTD\n\nAlso specify the company (APCo, JCCo, etc.) and any date range."
            },
            {
                "q": "Query the database",
                "a": "I need more specific information:\n\n1. What data are you looking for? (invoices, jobs, employees, etc.)\n2. Which module? (AP, AR, JC, GL, PR, etc.)\n3. Which company?\n4. What filters or conditions?\n\nPlease provide these details for an accurate query."
            },
            {
                "q": "Show me all the data",
                "a": "Viewpoint Vista contains thousands of tables across many modules. To help you, please specify:\n\n1. **Module:** AP, AR, JC, GL, PR, EM, etc.\n2. **Specific table or data type**\n3. **Company and date range**\n\nFor example: \"Show me AP invoices for company 1 from January 2024\""
            },
            {
                "q": "Write a report query",
                "a": "To write a report query, I need to know:\n\n1. **Report purpose:** What information should it show?\n2. **Data source:** Which module/tables?\n3. **Filters:** Company, date range, status?\n4. **Grouping/Aggregation:** Totals, summaries?\n\nAlso check if a vrv* or brv* reporting view already exists for your purpose."
            },
        ]

        for item in generic_questions:
            examples.append(NegativeExample(
                instruction=item["q"],
                output=item["a"]
            ))

        return examples

    def generate_case_sensitivity_examples(self) -> List[NegativeExample]:
        """Generate examples for case sensitivity errors."""
        examples = []

        case_errors = [
            ("apth", "APTH"),
            ("Apth", "APTH"),
            ("jcjm", "JCJM"),
            ("Jcjm", "JCJM"),
            ("apco", "APCo"),
            ("APCO", "APCo"),
            ("jcco", "JCCo"),
            ("JCCO", "JCCo"),
            ("vendorgroup", "VendorGroup"),
            ("VENDORGROUP", "VendorGroup"),
            ("invnum", "InvNum"),
            ("INVNUM", "InvNum"),
        ]

        for wrong, correct in case_errors:
            examples.append(NegativeExample(
                instruction=f"Query {wrong} table",
                output=f"The correct table/column name is '{correct}', not '{wrong}'.\n\nViewpoint uses Latin1_General_BIN collation which is case-sensitive. Always use the exact case as defined in the schema.\n\n```sql\nSELECT *\nFROM {correct} WITH (NOLOCK)\n```"
            ))

        return examples

    def save_dataset(self, examples: List[NegativeExample], output_path: str):
        """Save examples to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = [e.to_alpaca() for e in examples]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(examples)} negative examples to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate negative training examples")
    parser.add_argument('--vgpt2', type=str, default='C:/Github/VGPT2',
                        help='Path to VGPT2 repository')
    parser.add_argument('--output', type=str, default='data/vgpt2_v3/negative_examples.json',
                        help='Output file path')

    args = parser.parse_args()

    generator = NegativeExampleGenerator(args.vgpt2)
    examples = generator.generate_all()
    generator.save_dataset(examples, args.output)

    print(f"\nGenerated {len(examples)} negative examples to {args.output}")


if __name__ == "__main__":
    main()
