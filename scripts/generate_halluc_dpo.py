#!/usr/bin/env python3
"""
Generate hallucination-focused DPO pairs for VGPT2 v3.

This script creates DPO preference pairs that teach the model to:
1. Reject fake/non-existent table names
2. Reject fake column names on real tables  
3. Reject generic SQL patterns not in Vista
"""

import json
import random
from pathlib import Path

# Common fake table names people might use (not in Vista)
FAKE_TABLES = {
    # Invoice variations
    "Invoice": ("APTH", "AP Transaction Header"),
    "Invoices": ("APTH", "AP Transaction Header"),
    "InvoiceHeader": ("APTH", "AP Transaction Header"),
    "InvoiceDetail": ("APTL", "AP Transaction Line"),
    "InvoiceLine": ("APTL", "AP Transaction Line"),
    "APInvoice": ("APTH", "AP Transaction Header"),
    "APInvoiceHeader": ("APTH", "AP Transaction Header"),
    "VendorInvoice": ("APTH", "AP Transaction Header"),
    "ARInvoice": ("ARTH", "AR Transaction Header"),
    
    # Customer variations
    "Customer": ("ARCM", "AR Customer Master"),
    "Customers": ("ARCM", "AR Customer Master"),
    "CustomerMaster": ("ARCM", "AR Customer Master"),
    "CustomerInfo": ("ARCM", "AR Customer Master"),
    "ClientMaster": ("ARCM", "AR Customer Master"),
    "Client": ("ARCM", "AR Customer Master"),
    "Clients": ("ARCM", "AR Customer Master"),
    
    # Vendor variations
    "Vendor": ("APVM", "AP Vendor Master"),
    "Vendors": ("APVM", "AP Vendor Master"),
    "VendorMaster": ("APVM", "AP Vendor Master"),
    "Supplier": ("APVM", "AP Vendor Master"),
    "Suppliers": ("APVM", "AP Vendor Master"),
    
    # Employee variations
    "Employee": ("PREH", "PR Employee Header"),
    "Employees": ("PREH", "PR Employee Header"),
    "EmployeeMaster": ("PREH", "PR Employee Header"),
    "Staff": ("PREH", "PR Employee Header"),
    "Worker": ("PREH", "PR Employee Header"),
    "Workers": ("PREH", "PR Employee Header"),
    
    # Job/Project variations
    "Job": ("JCJM", "JC Job Master"),
    "Jobs": ("JCJM", "JC Job Master"),
    "JobMaster": ("JCJM", "JC Job Master"),
    "Project": ("JCJM", "JC Job Master"),
    "Projects": ("JCJM", "JC Job Master"),
    "ProjectMaster": ("JCJM", "JC Job Master"),
    
    # Work Order variations
    "WorkOrder": ("EMWH", "EM Work Order Header"),
    "WorkOrders": ("EMWH", "EM Work Order Header"),
    "WorkOrderHeader": ("EMWH", "EM Work Order Header"),
    
    # Timecard variations
    "Timecard": ("PRTH", "PR Timecard Header"),
    "Timecards": ("PRTH", "PR Timecard Header"),
    "TimeCard": ("PRTH", "PR Timecard Header"),
    "TimeCards": ("PRTH", "PR Timecard Header"),
    "TimecardHeader": ("PRTH", "PR Timecard Header"),
    "PayrollTimecard": ("PRTH", "PR Timecard Header"),
    
    # PO variations  
    "PurchaseOrder": ("POHD", "PO Header"),
    "PurchaseOrders": ("POHD", "PO Header"),
    "PO": ("POHD", "PO Header"),
    "POHeader": ("POHD", "PO Header"),
    
    # GL variations
    "Transaction": ("GLDT", "GL Detail"),
    "Transactions": ("GLDT", "GL Detail"),
    "GLTransaction": ("GLDT", "GL Detail"),
    "GLTransactions": ("GLDT", "GL Detail"),
    "JournalEntry": ("GLDT", "GL Detail"),
    "Account": ("GLAC", "GL Account"),
    "Accounts": ("GLAC", "GL Account"),
    "ChartOfAccounts": ("GLAC", "GL Account"),
    "GLAccount": ("GLAC", "GL Account"),
    
    # Equipment variations
    "Equipment": ("EMEM", "EM Equipment Master"),
    "EquipmentMaster": ("EMEM", "EM Equipment Master"),
    "Asset": ("EMEM", "EM Equipment Master"),
    "Assets": ("EMEM", "EM Equipment Master"),
    
    # Material/Inventory variations
    "Material": ("INMT", "IN Material"),
    "Materials": ("INMT", "IN Material"),
    "Inventory": ("INMT", "IN Material"),
    "InventoryItem": ("INMT", "IN Material"),
    
    # Contract variations
    "Contract": ("JCCM", "JC Contract Master"),
    "Contracts": ("JCCM", "JC Contract Master"),
    "ContractMaster": ("JCCM", "JC Contract Master"),
    
    # Subcontract variations
    "Subcontract": ("SLHD", "SL Header"),
    "Subcontracts": ("SLHD", "SL Header"),
    "SubcontractHeader": ("SLHD", "SL Header"),
    
    # Cost Code variations
    "CostCode": ("JCCH", "JC Cost Header"),
    "CostCodes": ("JCCH", "JC Cost Header"),
    "Phase": ("JCCH", "JC Cost Header"),
    "Phases": ("JCCH", "JC Cost Header"),
    
    # Change Order variations
    "ChangeOrder": ("JCOI", "JC Change Order Item"),
    "ChangeOrders": ("JCOI", "JC Change Order Item"),
    
    # Billing variations
    "Billing": ("ARBH", "AR Bill Header"),
    "BillingHeader": ("ARBH", "AR Bill Header"),
    "ARBilling": ("ARBH", "AR Bill Header"),
    
    # Generic/common fake names
    "User": ("DDUP", "DD User Profile"),
    "Users": ("DDUP", "DD User Profile"),
    "UserMaster": ("DDUP", "DD User Profile"),
    "Order": ("POHD", "PO Header"),
    "Orders": ("POHD", "PO Header"),
    "Payment": ("APTH", "AP Transaction Header"),
    "Payments": ("APTH", "AP Transaction Header"),
    "Company": ("HQCO", "HQ Company"),
    "Companies": ("HQCO", "HQ Company"),
    "Department": ("JCDP", "JC Department"),
    "Departments": ("JCDP", "JC Department"),
    "Location": ("HQLC", "HQ Location"),
    "Locations": ("HQLC", "HQ Location"),
}

# Question templates for fake tables
FAKE_TABLE_TEMPLATES = [
    "Query the {fake} table",
    "Select all from {fake}",
    "What columns are in {fake}?",
    "Describe the {fake} table",
    "Get all data from {fake}",
    "Show me the {fake} table structure",
    "What's in the {fake} table?",
    "List all records from {fake}",
    "SELECT * FROM {fake}",
    "How do I query {fake}?",
    "What fields does {fake} have?",
    "Get {fake} data",
    "Query {fake} for company 1",
    "Select from {fake} where company = 1",
    "Join {fake} with other tables",
    "Write SQL to query {fake}",
    "I need data from {fake}",
    "Show {fake} records",
    "What is the schema for {fake}?",
    "How is {fake} structured?",
]

# Fake columns that don't exist
FAKE_COLUMNS = {
    "APTH": [
        ("InvoiceID", "Use KeyID or APTrans instead"),
        ("InvoiceNumber", "Use InvNum instead"),
        ("InvoiceDate", "Use InvDate instead"),
        ("InvoiceAmount", "Use GrossAmt instead"),
        ("VendorID", "Use Vendor instead"),
        ("VendorName", "Join with APVM to get Name"),
        ("CustomerID", "APTH is for AP, not AR"),
        ("TotalAmount", "Use GrossAmt instead"),
        ("Status", "Use OpenYN or Status columns"),
        ("CreatedDate", "Use InvDate or AddedDate"),
        ("ModifiedDate", "Use the audit columns"),
    ],
    "ARCM": [
        ("CustomerID", "Use Customer instead"),
        ("CustomerName", "Use Name instead"),
        ("CustomerNumber", "Use Customer instead"),
        ("Email", "Use the contact fields"),
        ("Phone", "Use Phone1 or Phone2"),
        ("Address", "Use Address1, Address2"),
        ("City", "City exists - check column case"),
        ("ContactName", "Use Contact instead"),
        ("Balance", "Calculate from ARTH"),
    ],
    "APVM": [
        ("VendorID", "Use Vendor instead"),
        ("VendorName", "Use Name instead"),
        ("VendorNumber", "Use Vendor instead"),
        ("SupplierName", "Use Name instead"),
        ("Email", "Use the contact fields"),
        ("Phone", "Use Phone1 or Phone2"),
        ("ContactName", "Use Contact instead"),
    ],
    "PREH": [
        ("EmployeeID", "Use Employee instead"),
        ("EmployeeName", "Use FirstName, LastName"),
        ("FirstName", "FirstName exists - check case"),
        ("LastName", "LastName exists - check case"),
        ("Salary", "Use PRRate or check PREA"),
        ("HireDate", "Use HireDate - check column case"),
        ("Department", "Use Dept instead"),
        ("SSN", "Use SSN - check column case"),
    ],
    "JCJM": [
        ("JobID", "Use Job instead"),
        ("JobName", "Use Description instead"),
        ("JobNumber", "Use Job instead"),
        ("ProjectID", "Use Job instead"),
        ("ProjectName", "Use Description instead"),
        ("StartDate", "Check ActualStartDate"),
        ("EndDate", "Check ProjectedCloseDate"),
        ("Budget", "Calculate from JCCH"),
        ("Status", "Use JobStatus instead"),
        ("CustomerID", "Use Customer instead"),
    ],
    "GLDT": [
        ("TransactionID", "Use KeyID instead"),
        ("AccountNumber", "Use GLAcct instead"),
        ("Amount", "Use Amount - check case"),
        ("PostDate", "Use ActDate instead"),
        ("Description", "Use Description - check case"),
    ],
}

# Generic SQL patterns that don't apply to Vista
GENERIC_SQL_PATTERNS = [
    ("Join Users with Orders", "Neither 'Users' nor 'Orders' exist in Vista. For users, use DDUP. For purchase orders, use POHD."),
    ("Join Customers with Invoices", "'Customers' and 'Invoices' don't exist. Use ARCM for customers and ARTH for AR transactions, or APTH for AP."),
    ("Get UserID from Users table", "'Users' doesn't exist in Vista. User information is in DDUP (DD User Profile)."),
    ("Join Employee with Department", "Use PREH for employees. Department info is in PRDP or Dept column on PREH."),
    ("Select from Customer_Orders", "No 'Customer_Orders' table. Viewpoint uses module prefixes like AR, AP, JC."),
    ("Query the sales table", "'Sales' doesn't exist. For AR transactions use ARTH, for billing use ARBH."),
    ("Join Product with Category", "Neither 'Product' nor 'Category' exist. For materials use INMT."),
    ("Get data from OrderDetails", "'OrderDetails' doesn't exist. For PO details use POIT, for AP use APTL."),
    ("Query CustomerAddress", "'CustomerAddress' doesn't exist. Address fields are in ARCM."),
    ("Join Invoice with Payment", "Use APTH for AP invoices and payment info is tracked via Status."),
    ("Select from tbl_Customers", "Vista doesn't use 'tbl_' prefixes. Use ARCM for customers."),
    ("Query dbo.Invoices", "Vista uses module prefixes, not dbo.Invoices. Use APTH."),
    ("Join Organization with User", "'Organization' doesn't exist. Use HQCO for companies, DDUP for users."),
    ("Get all from Contacts", "'Contacts' doesn't exist. Contact info is embedded in ARCM, APVM, etc."),
    ("Query TimeEntry table", "'TimeEntry' doesn't exist. Use PRTH for timecards."),
    ("Select from BillingDetail", "'BillingDetail' doesn't exist. Use ARBL for AR billing lines."),
    ("Join Project with Task", "'Task' doesn't exist. Use JCCH for job phases/cost codes."),
    ("Query InventoryTransaction", "'InventoryTransaction' doesn't exist. Use INDT for inventory detail."),
    ("Get from PaymentHistory", "'PaymentHistory' doesn't exist. Payment tracking is in APTH."),
    ("Join Vendor with PurchaseOrder", "'PurchaseOrder' doesn't exist. Join APVM with POHD for vendor POs."),
]


def generate_fake_table_pairs():
    """Generate DPO pairs for fake table rejection."""
    pairs = []

    for fake_table, (real_table, description) in FAKE_TABLES.items():
        for template in FAKE_TABLE_TEMPLATES:
            question = template.format(fake=fake_table)

            # Chosen: Reject and suggest correct table
            chosen = f"There is no '{fake_table}' table in Viewpoint Vista. The correct table is {real_table} ({description}).\n\n```sql\nSELECT *\nFROM {real_table} WITH (NOLOCK)\nWHERE Co = @Co\n```"

            # Rejected: Generate SQL for fake table
            rejected = f"```sql\nSELECT *\nFROM {fake_table}\n```"

            pairs.append({
                "instruction": question,
                "input": "",
                "chosen": chosen,
                "rejected": rejected
            })

    return pairs


def generate_fake_column_pairs():
    """Generate DPO pairs for fake column rejection."""
    pairs = []

    templates = [
        "Get {col} from {table}",
        "Select {col} from {table}",
        "What is {col} in {table}?",
        "Query {table} for {col}",
        "SELECT {col} FROM {table}",
    ]

    for table, columns in FAKE_COLUMNS.items():
        for fake_col, suggestion in columns:
            for template in templates:
                question = template.format(col=fake_col, table=table)

                chosen = f"The column '{fake_col}' does not exist in {table}. {suggestion}\n\nUse proper Vista column names with exact casing (Latin1_General_BIN collation)."

                rejected = f"```sql\nSELECT {fake_col}\nFROM {table}\n```"

                pairs.append({
                    "instruction": question,
                    "input": "",
                    "chosen": chosen,
                    "rejected": rejected
                })

    return pairs


def generate_generic_sql_pairs():
    """Generate DPO pairs for generic SQL pattern rejection."""
    pairs = []

    for question, answer in GENERIC_SQL_PATTERNS:
        # Add variations
        variations = [
            question,
            f"Write SQL to {question.lower()}",
            f"How do I {question.lower()}?",
            f"I need to {question.lower()}",
        ]

        for q in variations:
            pairs.append({
                "instruction": q,
                "input": "",
                "chosen": answer,
                "rejected": f"```sql\n-- Invalid query for non-existent tables\nSELECT * FROM ...\n```"
            })

    return pairs


def main():
    print("Generating hallucination-focused DPO pairs...")

    # Generate all pairs
    fake_table_pairs = generate_fake_table_pairs()
    print(f"  Fake table pairs: {len(fake_table_pairs)}")

    fake_column_pairs = generate_fake_column_pairs()
    print(f"  Fake column pairs: {len(fake_column_pairs)}")

    generic_sql_pairs = generate_generic_sql_pairs()
    print(f"  Generic SQL pairs: {len(generic_sql_pairs)}")

    # Combine all
    all_pairs = fake_table_pairs + fake_column_pairs + generic_sql_pairs
    print(f"\nTotal hallucination pairs: {len(all_pairs)}")

    # Save to file
    output_path = Path("data/vgpt2_v3_dpo_halluc_raw.json")
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"Saved to: {output_path}")

    return all_pairs


if __name__ == "__main__":
    main()

