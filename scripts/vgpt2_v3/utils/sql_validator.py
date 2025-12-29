#!/usr/bin/env python3
"""
SQL Validator Utility for VGPT2 v3
==================================
Validates SQL queries against Viewpoint Vista conventions and schema.

This module checks:
- SQL syntax (basic parsing)
- Table/view existence
- Column name validity (case-sensitive)
- WITH (NOLOCK) usage
- Company column filtering
- Table alias violations

Usage:
    from utils.sql_validator import SQLValidator

    validator = SQLValidator("C:/Github/VGPT2")

    result = validator.validate("SELECT * FROM APTH WHERE APCo = 1")
    if result.is_valid:
        print("SQL is valid")
    else:
        print(f"Errors: {result.errors}")
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple
from pathlib import Path

from .schema_loader import SchemaLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """A single validation error."""
    code: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    location: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of SQL validation."""
    sql: str
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    tables_found: Set[str] = field(default_factory=set)
    columns_found: Set[str] = field(default_factory=set)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"{status}: {self.error_count} errors, {self.warning_count} warnings"


class SQLValidator:
    """
    Validates SQL queries against Viewpoint Vista standards.

    Checks for:
    - Table/view existence
    - Column validity
    - WITH (NOLOCK) requirement
    - Company column filtering
    - No table aliases rule
    - Case sensitivity
    """

    # Patterns for SQL parsing
    TABLE_PATTERN = re.compile(
        r'(?:FROM|JOIN|INTO|UPDATE|DELETE\s+FROM)\s+(\[?[\w\.]+\]?)',
        re.IGNORECASE
    )

    COLUMN_PATTERN = re.compile(
        r'(?:SELECT|WHERE|AND|OR|ON|SET|ORDER\s+BY|GROUP\s+BY)\s+[\w\.\[\]]+',
        re.IGNORECASE
    )

    ALIAS_PATTERN = re.compile(
        r'(?:FROM|JOIN)\s+(\[?[\w\.]+\]?)\s+(?:AS\s+)?([a-zA-Z]\w*)\s+(?:WITH|ON|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|,|$)',
        re.IGNORECASE
    )

    NOLOCK_PATTERN = re.compile(
        r'(\[?[\w\.]+\]?)\s+WITH\s*\(\s*NOLOCK\s*\)',
        re.IGNORECASE
    )

    def __init__(self, vgpt2_path: str):
        """
        Initialize the validator.

        Args:
            vgpt2_path: Path to VGPT2 repository for schema lookups
        """
        self.schema = SchemaLoader(vgpt2_path)

        # Common non-table keywords that look like tables
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'IS',
            'NULL', 'JOIN', 'ON', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
            'GROUP', 'BY', 'ORDER', 'HAVING', 'UNION', 'ALL', 'AS',
            'WITH', 'NOLOCK', 'INSERT', 'UPDATE', 'DELETE', 'INTO',
            'VALUES', 'SET', 'CREATE', 'ALTER', 'DROP', 'TABLE', 'VIEW',
            'INDEX', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'DISTINCT',
            'TOP', 'LIKE', 'BETWEEN', 'EXISTS', 'ASC', 'DESC', 'LIMIT',
            'OFFSET', 'FETCH', 'NEXT', 'ROWS', 'ONLY', 'OVER', 'PARTITION',
            'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE', 'LAG', 'LEAD',
            'FIRST_VALUE', 'LAST_VALUE', 'SUM', 'COUNT', 'AVG', 'MIN', 'MAX',
            'COALESCE', 'ISNULL', 'NULLIF', 'CAST', 'CONVERT', 'DATEADD',
            'DATEDIFF', 'GETDATE', 'GETUTCDATE', 'YEAR', 'MONTH', 'DAY',
            'CTE', 'RECURSIVE', 'CROSS', 'APPLY', 'PIVOT', 'UNPIVOT'
        }

    def validate(self, sql: str, strict: bool = True) -> ValidationResult:
        """
        Validate a SQL query.

        Args:
            sql: SQL query to validate
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(sql=sql, is_valid=True)

        # Basic syntax check
        self._check_basic_syntax(sql, result)

        # Extract and validate tables
        tables = self._extract_tables(sql)
        result.tables_found = tables
        self._validate_tables(tables, sql, result)

        # Check WITH (NOLOCK)
        self._check_nolock(sql, tables, result)

        # Check for table aliases
        self._check_aliases(sql, result)

        # Check company column filtering
        self._check_company_filter(sql, tables, result)

        # Determine overall validity
        result.is_valid = len(result.errors) == 0
        if strict:
            result.is_valid = result.is_valid and len(result.warnings) == 0

        return result

    def _check_basic_syntax(self, sql: str, result: ValidationResult):
        """Check basic SQL syntax."""
        sql_upper = sql.upper().strip()

        # Must start with valid keyword
        valid_starts = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE', 'ALTER', 'DROP', '--']
        if not any(sql_upper.startswith(kw) for kw in valid_starts):
            result.errors.append(ValidationError(
                code='INVALID_START',
                message='SQL must start with SELECT, INSERT, UPDATE, DELETE, WITH, or DDL statement',
                severity='error'
            ))

        # Check balanced parentheses
        if sql.count('(') != sql.count(')'):
            result.errors.append(ValidationError(
                code='UNBALANCED_PARENS',
                message='Unbalanced parentheses in SQL',
                severity='error'
            ))

        # Check for common Oracle syntax errors
        if ' INTERVAL ' in sql_upper and ' YEAR' in sql_upper:
            result.errors.append(ValidationError(
                code='ORACLE_SYNTAX',
                message="Oracle INTERVAL syntax not valid in SQL Server. Use DATEADD() instead.",
                severity='error'
            ))

        if 'ROWNUM' in sql_upper:
            result.errors.append(ValidationError(
                code='ORACLE_SYNTAX',
                message="Oracle ROWNUM not valid in SQL Server. Use TOP or ROW_NUMBER() instead.",
                severity='error'
            ))

    def _extract_tables(self, sql: str) -> Set[str]:
        """Extract table names from SQL."""
        tables = set()

        matches = self.TABLE_PATTERN.findall(sql)
        for match in matches:
            # Clean up table name
            table = match.strip('[]').split('.')[-1]  # Remove schema prefix

            # Skip if it's a keyword
            if table.upper() in self.sql_keywords:
                continue

            # Skip if it looks like a CTE reference
            if table.lower().startswith('cte_'):
                continue

            tables.add(table)

        return tables

    def _validate_tables(self, tables: Set[str], sql: str, result: ValidationResult):
        """Validate that tables exist in schema."""
        for table in tables:
            if not self.schema.table_exists(table):
                suggestions = self.schema.suggest_similar_tables(table)
                msg = f"Table/view '{table}' does not exist in Viewpoint Vista"
                if suggestions:
                    msg += f". Did you mean: {', '.join(suggestions[:3])}"

                result.errors.append(ValidationError(
                    code='TABLE_NOT_FOUND',
                    message=msg,
                    severity='error',
                    location=table
                ))

            # Check for base table usage in SELECT
            if table.startswith('b') and len(table) > 2:
                view_name = table[1:]  # Remove 'b' prefix
                if self.schema.table_exists(view_name):
                    sql_upper = sql.upper()
                    if 'SELECT' in sql_upper and 'INSERT' not in sql_upper and 'UPDATE' not in sql_upper:
                        result.warnings.append(ValidationError(
                            code='BASE_TABLE_IN_SELECT',
                            message=f"Use view '{view_name}' instead of base table '{table}' for SELECT queries",
                            severity='warning',
                            location=table
                        ))

    def _check_nolock(self, sql: str, tables: Set[str], result: ValidationResult):
        """Check that WITH (NOLOCK) is used for all tables in SELECT queries."""
        if 'SELECT' not in sql.upper():
            return

        # Find tables with NOLOCK
        nolock_matches = self.NOLOCK_PATTERN.findall(sql)
        tables_with_nolock = {m.strip('[]').split('.')[-1] for m in nolock_matches}

        # Check each table
        for table in tables:
            if table not in tables_with_nolock:
                result.warnings.append(ValidationError(
                    code='MISSING_NOLOCK',
                    message=f"Table '{table}' should have WITH (NOLOCK) for SELECT queries",
                    severity='warning',
                    location=table
                ))

    def _check_aliases(self, sql: str, result: ValidationResult):
        """Check for table alias usage (not allowed in Viewpoint)."""
        matches = self.ALIAS_PATTERN.findall(sql)

        for table, alias in matches:
            # Skip if alias looks like a keyword or is the same as table
            if alias.upper() in self.sql_keywords:
                continue
            if alias.upper() == table.upper().strip('[]'):
                continue

            result.warnings.append(ValidationError(
                code='TABLE_ALIAS',
                message=f"Table alias '{alias}' for '{table}' violates Viewpoint standards. Use full table name.",
                severity='warning',
                location=f"{table} {alias}"
            ))

    def _check_company_filter(self, sql: str, tables: Set[str], result: ValidationResult):
        """Check that company columns are filtered."""
        if 'SELECT' not in sql.upper():
            return

        for table in tables:
            company_col = self.schema.get_company_column(table)
            if not company_col:
                continue

            # Check if company column is in WHERE or JOIN
            if company_col not in sql and company_col.lower() not in sql.lower():
                result.warnings.append(ValidationError(
                    code='MISSING_COMPANY_FILTER',
                    message=f"Consider filtering by {company_col} for table '{table}' to ensure data isolation",
                    severity='warning',
                    location=table
                ))

    def validate_batch(self, queries: List[str]) -> List[ValidationResult]:
        """Validate multiple queries."""
        return [self.validate(q) for q in queries]

    def get_stats(self, results: List[ValidationResult]) -> dict:
        """Get statistics from validation results."""
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        errors = sum(r.error_count for r in results)
        warnings = sum(r.warning_count for r in results)

        # Count error types
        error_types = {}
        for r in results:
            for e in r.errors + r.warnings:
                error_types[e.code] = error_types.get(e.code, 0) + 1

        return {
            'total': total,
            'valid': valid,
            'invalid': total - valid,
            'valid_percent': round(100 * valid / total, 1) if total > 0 else 0,
            'total_errors': errors,
            'total_warnings': warnings,
            'error_types': error_types
        }


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SQL Validator Utility")
    parser.add_argument('--vgpt2', type=str, default='C:/Github/VGPT2',
                        help='Path to VGPT2 repository')
    parser.add_argument('--sql', type=str, help='SQL query to validate')
    parser.add_argument('--file', type=str, help='File containing SQL to validate')

    args = parser.parse_args()

    validator = SQLValidator(args.vgpt2)

    if args.sql:
        result = validator.validate(args.sql)
        print(f"\nValidation Result: {result.summary()}")
        print(f"Tables found: {result.tables_found}")

        if result.errors:
            print("\nErrors:")
            for e in result.errors:
                print(f"  [{e.code}] {e.message}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  [{w.code}] {w.message}")

    elif args.file:
        with open(args.file, 'r') as f:
            sql = f.read()
        result = validator.validate(sql)
        print(f"\nValidation Result: {result.summary()}")

    else:
        # Demo
        test_queries = [
            "SELECT * FROM APTH WITH (NOLOCK) WHERE APCo = 1",
            "SELECT * FROM Invoice WHERE Paid = 0",
            "SELECT a.* FROM APTH a WHERE a.APCo = 1",
            "SELECT * FROM bAPTH WHERE APCo = 1",
        ]

        print("\nSQL Validator Demo")
        print("=" * 60)

        for sql in test_queries:
            result = validator.validate(sql)
            print(f"\nSQL: {sql[:50]}...")
            print(f"Result: {result.summary()}")
