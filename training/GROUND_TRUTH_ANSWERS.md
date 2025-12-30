# Ground Truth Answers for VGPT2 v3 Validation

**Created:** 2025-12-30
**Purpose:** Reference answers for all 47 test questions

---

## Schema Questions (10)

### schema_001: What columns are in the APTH table?
**Expected Keywords:** APCo, Mth, APTrans, Vendor, InvNum

**Ground Truth:**
APTH (AP Transaction Header) contains 61 columns. Key columns:
- **APCo** (tinyint) - AP Company
- **Mth** (smalldatetime) - Batch Month
- **APTrans** (int) - Transaction Number
- **VendorGroup** (tinyint) - Vendor Group for cross-company vendor lookup
- **Vendor** (int) - Vendor Number
- **InvId** (varchar) - Invoice ID (internal identifier)
- **APRef** (varchar) - AP Reference (often vendor's invoice number)
- **Description** (varchar) - Invoice description
- **InvDate** (smalldatetime) - Invoice Date
- **InvTotal** (numeric) - Invoice Total Amount
- **Status** (tinyint) - 0=Open, 1=Partially Paid, 2=Fully Paid

---

### schema_002: What is the primary key of JCJM?
**Expected Keywords:** JCCo, Job

**Ground Truth:**
The primary key of JCJM (Job Cost Job Master) is the composite key:
- **JCCo** (tinyint) - Job Cost Company
- **Job** (varchar) - Job Number

These two columns together uniquely identify each job.

---

### schema_003: Describe the APTL table structure
**Expected Keywords:** APCo, Mth, APTrans, APLine

**Ground Truth:**
APTL (AP Transaction Line) contains 76 columns. It stores line item details for AP invoices.
Primary key: APCo, Mth, APTrans, APLine (links to APTH + line number)

Key columns:
- **APCo, Mth, APTrans** - Links to parent APTH
- **APLine** (smallint) - Line sequence number
- **LineType** - Type of line (expense, PO, subcontract, etc.)
- **GrossAmt** - Line amount
- **JCCo, Job, Phase** - Job cost distribution
- **GLCo, GLAcct** - GL account for expense

---

### schema_004: What data type is the Vendor column in APTH?
**Expected Keywords:** int, integer, numeric

**Ground Truth:**
The Vendor column in APTH is **int** (integer). It's a numeric identifier that links to APVM.Vendor.

---

### schema_005: What columns link APTH to APTL?
**Expected Keywords:** APCo, Mth, APTrans

**Ground Truth:**
Three columns link APTH (header) to APTL (lines):
- **APCo** - AP Company
- **Mth** - Batch Month
- **APTrans** - Transaction Number

JOIN: `APTH.APCo = APTL.APCo AND APTH.Mth = APTL.Mth AND APTH.APTrans = APTL.APTrans`

---

### schema_006: What is the difference between APTH and bAPTH?
**Expected Keywords:** view, base, table, SELECT

**Ground Truth:**
- **bAPTH** is the base **table** that stores the actual data
- **APTH** is a **view** that wraps bAPTH

Always use the view (APTH) for queries. The base tables (prefixed with 'b') should not be directly queried in production code as they may lack business logic implemented in the views.

---

### schema_007: List all tables in the AP module
**Expected Keywords:** APTH, APTL, APVM

**Ground Truth:**
Key tables in the AP (Accounts Payable) module:
- **APTH** - Transaction Header (invoices)
- **APTL** - Transaction Line (invoice lines)
- **APVM** - Vendor Master
- **APTB** - Transaction Batch
- **APCM** - Check Master (payments)
- **APCD** - Check Detail
- **APHB** - Hold Batch
- **APCO** - Company settings

---

### schema_008: What is VendorGroup in Viewpoint?
**Expected Keywords:** group, master, shared, company

**Ground Truth:**
VendorGroup is a mechanism for sharing vendors across multiple companies. Instead of each company maintaining separate vendor records, vendors are defined at a VendorGroup level and can be used by any company assigned to that group.

APVM is keyed by (VendorGroup, Vendor) - the same Vendor number in different VendorGroups are different vendors.

---

### schema_009: What company column does JCCD use?
**Expected Keywords:** JCCo

### sql_001: Write SQL to get all unpaid AP invoices for company 1
**Expected Keywords:** SELECT, FROM, APTH, WITH, NOLOCK, APCo, Status
**Forbidden Keywords:** Invoice, Invoices

**Ground Truth:**
```sql
SELECT *
FROM APTH WITH (NOLOCK)
WHERE APCo = 1
  AND Status = 0
```
Note: Status = 0 means Open/Unpaid in Viewpoint.

---

### sql_002: Query all active jobs for JC company 5
**Expected Keywords:** SELECT, JCJM, WITH, NOLOCK, JCCo, 5

**Ground Truth:**
```sql
SELECT *
FROM JCJM WITH (NOLOCK)
WHERE JCCo = 5
  AND JobStatus = 1
```
Note: JobStatus = 1 typically means Active.

---

### sql_003: Get vendor name and total invoiced amount
**Expected Keywords:** SELECT, APVM, APTH, JOIN, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT APVM.Name, SUM(APTH.InvTotal) AS TotalInvoiced
FROM APTH WITH (NOLOCK)
INNER JOIN APVM WITH (NOLOCK)
    ON APTH.VendorGroup = APVM.VendorGroup
    AND APTH.Vendor = APVM.Vendor
GROUP BY APVM.Name
```

---

### sql_004: Find AP invoices over $10,000
**Expected Keywords:** SELECT, APTH, GrossAmt, 10000, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT *
FROM APTH WITH (NOLOCK)
WHERE InvTotal > 10000
```
Note: InvTotal contains the invoice total amount.

---

### sql_005: List all GL accounts for company 1
**Expected Keywords:** SELECT, GLAC, GLCo, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT *
FROM GLAC WITH (NOLOCK)
WHERE GLCo = 1
```

---

### sql_006: Get payroll timecards for employee 100
**Expected Keywords:** SELECT, PRTH, Employee, 100, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT *
FROM PRTH WITH (NOLOCK)
WHERE Employee = 100
```

---

### sql_007: Query AR transactions for customer group 1
**Expected Keywords:** SELECT, ARTH, CustGroup, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT *
FROM ARTH WITH (NOLOCK)
WHERE CustGroup = 1
```

---

### sql_008: Get job costs by phase
**Expected Keywords:** SELECT, JCCD, Phase, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT Job, Phase, SUM(ActualCost) AS TotalCost
FROM JCCD WITH (NOLOCK)
GROUP BY Job, Phase
```

---

### sql_009: List equipment for EM company 1
**Expected Keywords:** SELECT, EMEM, EMCo, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT *
FROM EMEM WITH (NOLOCK)
WHERE EMCo = 1
```

---

### sql_010: Get purchase orders pending approval
**Expected Keywords:** SELECT, POHD, Status, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT *
FROM POHD WITH (NOLOCK)
WHERE Status = 0
```
Note: Check actual Status values as they may vary by implementation.

---

### sql_011: Find all invoices from January 2024
**Expected Keywords:** SELECT, APTH, Mth, 2024-01-01, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT *
FROM APTH WITH (NOLOCK)
WHERE Mth = '2024-01-01'
```
Note: Mth in Viewpoint is always the first day of the month.

---

### sql_012: Get subcontract commitments for a job
**Expected Keywords:** SELECT, SLHD, Job, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT *
FROM SLHD WITH (NOLOCK)
WHERE Job = @Job
```

---

## Hallucination Questions (10)

These questions ask about FAKE tables that do NOT exist in Viewpoint Vista.
The model MUST refuse and suggest the correct Vista table.

### halluc_001: What columns are in the Invoice table?
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist, no table, APTH

**Ground Truth:**
There is no "Invoice" table in Viewpoint Vista. For invoice data, use:
- **APTH** - AP Transaction Header (AP invoices/bills)
- **ARTH** - AR Transaction Header (AR invoices to customers)

---

### halluc_002: Query the Customers table
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist, ARCM

**Ground Truth:**
There is no "Customers" table in Viewpoint Vista. Customer data is stored in:
- **ARCM** - AR Customer Master

---

### halluc_003: What is the UserPreferences table?
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist, no table

**Ground Truth:**
There is no "UserPreferences" table in Viewpoint Vista. This is a generic table name that doesn't exist in the Vista schema.

---

### halluc_004: Describe the Payments table structure
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist, APCM, check

**Ground Truth:**
There is no "Payments" table in Viewpoint Vista. Payment data is stored in:
- **APCM** - AP Check Master (vendor payments)
- **ARCD** - AR Cash Detail (customer payments received)

---

### halluc_005: How do I join Invoice and Customer?
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist

**Ground Truth:**
Neither "Invoice" nor "Customer" tables exist in Viewpoint Vista.
For AR invoices and customers, use: ARTH JOIN ARCM ON CustGroup and Customer
For AP invoices and vendors, use: APTH JOIN APVM ON VendorGroup and Vendor

---

### halluc_006: What columns are in the SalesData table?
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist

**Ground Truth:**
There is no "SalesData" table in Viewpoint Vista. This is a generic analytics table name not used in Vista.

---

### halluc_007: Query the Orders table for pending orders
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist, POHD

**Ground Truth:**
There is no "Orders" table in Viewpoint Vista. For purchase orders, use:
- **POHD** - PO Header

---

### halluc_008: What is the EmployeeMaster table?
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist, PREH

**Ground Truth:**
There is no "EmployeeMaster" table in Viewpoint Vista. Employee data is stored in:
- **PREH** - PR Employee Header (main employee master)

---

### halluc_009: Describe the TransactionHistory table
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist

**Ground Truth:**
There is no "TransactionHistory" table in Viewpoint Vista. Transactions are stored in module-specific tables like APTH, ARTH, GLDT, JCCD, etc.

---

### halluc_010: How do I query the Project table?
**expect_refusal:** TRUE
**Expected Keywords:** not exist, doesn't exist, JCJM

**Ground Truth:**
There is no "Project" table in Viewpoint Vista. Project/Job data is stored in:
- **JCJM** - JC Job Master

---

## JOIN Questions (5)

### join_001: How do I join APTH and APTL?
**Expected Keywords:** APCo, Mth, APTrans, JOIN, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT APTH.*, APTL.*
FROM APTH WITH (NOLOCK)
INNER JOIN APTL WITH (NOLOCK)
    ON APTH.APCo = APTL.APCo
    AND APTH.Mth = APTL.Mth
    AND APTH.APTrans = APTL.APTrans
```
The key is 3 columns: APCo, Mth, APTrans.

---

### join_002: How do I join APTH with APVM?
**Expected Keywords:** VendorGroup, Vendor, JOIN, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT APTH.*, APVM.Name
FROM APTH WITH (NOLOCK)
INNER JOIN APVM WITH (NOLOCK)
    ON APTH.VendorGroup = APVM.VendorGroup
    AND APTH.Vendor = APVM.Vendor
```
Must join on BOTH VendorGroup AND Vendor.

---

### join_003: How do I join JCJM with JCCD?
**Expected Keywords:** JCCo, Job, JOIN, WITH, NOLOCK

**Ground Truth:**
```sql
SELECT JCJM.*, JCCD.*
FROM JCJM WITH (NOLOCK)
INNER JOIN JCCD WITH (NOLOCK)
    ON JCJM.JCCo = JCCD.JCCo
    AND JCJM.Job = JCCD.Job
```

---

### join_004: What columns link ARTH to ARCM?
**Expected Keywords:** CustGroup, Customer

**Ground Truth:**
ARTH links to ARCM on:
- **CustGroup** - Customer Group
- **Customer** - Customer Number

Both columns are required for the join.

---

### join_005: How do I join PRTH with PREH?
**Expected Keywords:** PRCo, Employee, JOIN

**Ground Truth:**
```sql
SELECT PRTH.*, PREH.LastName, PREH.FirstName
FROM PRTH WITH (NOLOCK)
INNER JOIN PREH WITH (NOLOCK)
    ON PRTH.PRCo = PREH.PRCo
    AND PRTH.Employee = PREH.Employee
```

---

## Error Correction Questions (5)

### error_001: Fix this query: SELECT * FROM bAPTH WHERE APCo = 1
**Expected Keywords:** APTH, WITH, NOLOCK, view

**Ground Truth:**
Issues:
1. Uses base table (bAPTH) instead of view (APTH)
2. Missing WITH (NOLOCK)

Fixed:
```sql
SELECT * FROM APTH WITH (NOLOCK) WHERE APCo = 1
```

---

### error_002: Fix this query: SELECT apco, vendor FROM APTH
**Expected Keywords:** APCo, Vendor, case

**Ground Truth:**
Issues:
1. Incorrect case - should be APCo, Vendor (PascalCase)
2. Missing WITH (NOLOCK)

Fixed:
```sql
SELECT APCo, Vendor FROM APTH WITH (NOLOCK)
```

---

### error_003: Fix this query: SELECT a.* FROM APTH a WHERE a.APCo = 1
**Expected Keywords:** alias, full, table, name

**Ground Truth:**
Issues:
1. Uses table alias (a) - Vista convention is to use full table names
2. Missing WITH (NOLOCK)

Fixed:
```sql
SELECT APTH.* FROM APTH WITH (NOLOCK) WHERE APTH.APCo = 1
```

---

### error_004: Fix this query: SELECT * FROM APTH WHERE Mth = '2024-01'
**Expected Keywords:** 2024-01-01, first, day, month

**Ground Truth:**
Issues:
1. Mth must be first day of month: '2024-01-01' not '2024-01'
2. Missing WITH (NOLOCK)

Fixed:
```sql
SELECT * FROM APTH WITH (NOLOCK) WHERE Mth = '2024-01-01'
```

---

### error_005: Fix this query: SELECT * FROM APTH JOIN APTL ON APTH.APTrans = APTL.APTrans
**Expected Keywords:** APCo, Mth, incomplete, JOIN

**Ground Truth:**
Issues:
1. Incomplete JOIN - needs APCo AND Mth AND APTrans
2. Missing WITH (NOLOCK) on both tables

Fixed:
```sql
SELECT *
FROM APTH WITH (NOLOCK)
JOIN APTL WITH (NOLOCK)
    ON APTH.APCo = APTL.APCo
    AND APTH.Mth = APTL.Mth
    AND APTH.APTrans = APTL.APTrans
```

---

## Business Logic Questions (5)

### biz_001: What status codes indicate an unpaid AP invoice?
**Expected Keywords:** 0, Open, Status

**Ground Truth:**
In APTH, the Status column indicates payment status:
- **0** = Open (unpaid)
- **1** = Partially Paid
- **2** = Fully Paid

---

### biz_002: How does batch processing work in AP?
**Expected Keywords:** batch, APTB, post, header

**Ground Truth:**
AP uses batch processing:
1. Transactions are entered into **APTB** (AP Transaction Batch)
2. Batches are validated and posted
3. Posted transactions move to **APTH** (header) and **APTL** (lines)
4. BatchId tracks which batch a transaction came from

---

### biz_003: What is the difference between ActualCost and OrigEstCost in JCCD?
**Expected Keywords:** actual, original, estimate, budget

**Ground Truth:**
In JCCD (Job Cost Cost Detail):
- **OrigEstCost** - Original budgeted/estimated cost (what was planned)
- **ActualCost** - Actual incurred cost (what was spent)

The difference reveals cost variances.

---

### biz_004: How are vendors shared across companies?
**Expected Keywords:** VendorGroup, shared, company, master

**Ground Truth:**
Vendors are shared via VendorGroup:
1. APVM stores vendors at the VendorGroup level
2. Each company is assigned a VendorGroup
3. Multiple companies can share the same VendorGroup
4. This allows the same vendor master to be used across companies

---

### biz_005: When should I use vrv* views vs custom SQL?
**Expected Keywords:** report, Crystal, optimized, validated

**Ground Truth:**
The vrv* views (Vista Report Views) should be used when:
- Building Crystal Reports
- Need pre-validated, optimized queries
- Want consistent reporting across the organization

Custom SQL is appropriate when:
- Building one-off queries
- Need flexibility beyond what vrv* provides
- Doing development/testing

---

*Ground Truth Document Complete*

---

### schema_010: What columns are in GLAC?
**Expected Keywords:** GLCo, GLAcct, Description

**Ground Truth:**
GLAC (GL Account) contains 23 columns:
- **GLCo** (tinyint) - GL Company
- **GLAcct** (varchar) - GL Account Number
- **Description** (varchar) - Account description
- **AcctType** (char) - A=Asset, L=Liability, C=Capital, R=Revenue, E=Expense
- **SubType** - Sub-type for reporting
- **Active** (char) - Y/N flag
- **NormBal** - Normal balance (D=Debit, C=Credit)

---

## SQL Generation Questions (12)

