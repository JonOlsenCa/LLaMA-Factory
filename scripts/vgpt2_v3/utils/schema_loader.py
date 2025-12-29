#!/usr/bin/env python3
"""
Schema Loader Utility for VGPT2 v3
===================================
Loads and indexes Viewpoint Vista schema data for fast lookups.

This module provides:
- Column name validation
- Table/view existence checks
- Foreign key relationship lookups
- Module-to-company column mapping

Usage:
    from utils.schema_loader import SchemaLoader

    schema = SchemaLoader("C:/Github/VGPT2")

    # Check if a table exists
    schema.table_exists("APTH")  # True
    schema.table_exists("Invoice")  # False

    # Get columns for a table
    columns = schema.get_columns("APTH")

    # Validate a column name
    schema.column_exists("APTH", "APCo")  # True
    schema.column_exists("APTH", "apco")  # False (case-sensitive!)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    data_type: str
    is_nullable: bool
    table_name: str
    schema_name: str
    ordinal_position: int
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    default_value: Optional[str] = None


@dataclass
class TableInfo:
    """Information about a database table or view."""
    name: str
    schema_name: str
    object_type: str  # 'TABLE', 'VIEW'
    module: str
    columns: List[ColumnInfo] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.schema_name}.{self.name}" if self.schema_name != 'dbo' else self.name

    @property
    def column_names(self) -> Set[str]:
        return {col.name for col in self.columns}


@dataclass
class ForeignKeyInfo:
    """Information about a foreign key relationship."""
    parent_table: str
    parent_columns: List[str]
    child_table: str
    child_columns: List[str]
    constraint_name: str


class SchemaLoader:
    """
    Loads and indexes Viewpoint Vista schema data.

    Provides fast lookups for:
    - Table/view existence
    - Column validation
    - Foreign key relationships
    - Module mappings
    """

    # Module to company column mapping
    MODULE_COMPANY_COLUMNS = {
        'AP': 'APCo',
        'AR': 'ARCo',
        'GL': 'GLCo',
        'JC': 'JCCo',
        'PR': 'PRCo',
        'PM': 'PMCo',
        'EM': 'EMCo',
        'IN': 'INCo',
        'SM': 'SMCo',
        'HR': 'HRCo',
        'HQ': 'HQCo',
        'MS': 'MSCo',
        'PO': 'POCo',
        'SL': 'SLCo',
        'VA': None,  # Attachments don't have company
    }

    def __init__(self, vgpt2_path: str):
        """
        Initialize the schema loader.

        Args:
            vgpt2_path: Path to VGPT2 repository
        """
        self.vgpt2 = Path(vgpt2_path)
        self.metadata_dir = self.vgpt2 / "Viewpoint_Database" / "_Metadata"

        # Indexes (populated on first access)
        self._tables: Dict[str, TableInfo] = {}
        self._columns_by_table: Dict[str, Dict[str, ColumnInfo]] = {}
        self._foreign_keys: List[ForeignKeyInfo] = []
        self._fk_by_table: Dict[str, List[ForeignKeyInfo]] = {}
        self._all_table_names: Set[str] = set()
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load schema data on first access."""
        if not self._loaded:
            self._load_all()
            self._loaded = True

    def _load_all(self):
        """Load all schema data files."""
        logger.info("Loading Viewpoint schema data...")
        self._load_columns()
        self._load_tables_list()
        self._load_foreign_keys()
        logger.info(f"Loaded {len(self._tables)} tables/views, {sum(len(t.columns) for t in self._tables.values())} columns")

    def _load_columns(self):
        """Load columns.json and build table structures."""
        columns_file = self.metadata_dir / "columns.json"
        if not columns_file.exists():
            logger.warning(f"columns.json not found at {columns_file}")
            return

        with open(columns_file, 'r', encoding='utf-8') as f:
            columns_data = json.load(f)

        for col in columns_data:
            if not isinstance(col, dict):
                continue

            obj_name = col.get('ObjectName', '')
            schema_name = col.get('SchemaName', 'dbo')
            obj_type = col.get('ObjectType', 'TABLE')
            module = self._infer_module(obj_name)

            # Create table entry if needed
            if obj_name not in self._tables:
                self._tables[obj_name] = TableInfo(
                    name=obj_name,
                    schema_name=schema_name,
                    object_type=obj_type,
                    module=module
                )
                self._columns_by_table[obj_name] = {}

            # Create column info
            col_info = ColumnInfo(
                name=col.get('ColumnName', ''),
                data_type=col.get('DataType', 'unknown'),
                is_nullable=col.get('IsNullable', 'True') == 'True',
                table_name=obj_name,
                schema_name=schema_name,
                ordinal_position=int(col.get('OrdinalPosition', 0)),
                max_length=col.get('MaxLength'),
                precision=col.get('Precision'),
                scale=col.get('Scale'),
                default_value=col.get('DefaultValue')
            )

            self._tables[obj_name].columns.append(col_info)
            self._columns_by_table[obj_name][col_info.name] = col_info

    def _load_tables_list(self):
        """Load complete tables/views list."""
        tables_file = self.metadata_dir / "_Viewpoint_ALL_Views_Tables_Complete.json"
        if tables_file.exists():
            with open(tables_file, 'r', encoding='utf-8') as f:
                tables_data = json.load(f)

            for item in tables_data:
                name = item.get('name', item.get('TABLE_NAME', ''))
                if name:
                    self._all_table_names.add(name)
                    self._all_table_names.add(name.upper())
                    self._all_table_names.add(name.lower())

        # Also add from columns data
        self._all_table_names.update(self._tables.keys())

    def _load_foreign_keys(self):
        """Load foreign key relationships."""
        fk_file = self.metadata_dir / "foreign_keys.json"
        if not fk_file.exists():
            logger.warning(f"foreign_keys.json not found at {fk_file}")
            return

        with open(fk_file, 'r', encoding='utf-8') as f:
            fk_data = json.load(f)

        for fk in fk_data:
            if not isinstance(fk, dict):
                continue

            fk_info = ForeignKeyInfo(
                parent_table=fk.get('ParentTable', fk.get('parent_table', '')),
                parent_columns=fk.get('ParentColumns', fk.get('parent_columns', [])),
                child_table=fk.get('ChildTable', fk.get('child_table', '')),
                child_columns=fk.get('ChildColumns', fk.get('child_columns', [])),
                constraint_name=fk.get('ConstraintName', fk.get('constraint_name', ''))
            )

            self._foreign_keys.append(fk_info)

            # Index by both tables
            if fk_info.parent_table not in self._fk_by_table:
                self._fk_by_table[fk_info.parent_table] = []
            self._fk_by_table[fk_info.parent_table].append(fk_info)

            if fk_info.child_table not in self._fk_by_table:
                self._fk_by_table[fk_info.child_table] = []
            self._fk_by_table[fk_info.child_table].append(fk_info)

    def _infer_module(self, table_name: str) -> str:
        """Infer module from table name prefix."""
        if not table_name:
            return 'Unknown'

        # Handle b-prefixed tables
        name = table_name
        if name.startswith('b') and len(name) > 2:
            name = name[1:]

        # Check common prefixes
        for prefix in ['AP', 'AR', 'GL', 'JC', 'PR', 'PM', 'EM', 'IN', 'SM', 'HR', 'HQ', 'MS', 'PO', 'SL', 'VA']:
            if name.upper().startswith(prefix):
                return prefix

        return 'Unknown'

    # =========================================================================
    # Public API
    # =========================================================================

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table or view exists.

        Args:
            table_name: Table name to check (case-sensitive for exact match)

        Returns:
            True if table exists, False otherwise
        """
        self._ensure_loaded()
        return table_name in self._tables or table_name in self._all_table_names

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """
        Check if a column exists in a table (case-sensitive).

        Args:
            table_name: Table name
            column_name: Column name (case-sensitive!)

        Returns:
            True if column exists with exact case, False otherwise
        """
        self._ensure_loaded()
        if table_name not in self._columns_by_table:
            return False
        return column_name in self._columns_by_table[table_name]

    def get_table(self, table_name: str) -> Optional[TableInfo]:
        """Get table information."""
        self._ensure_loaded()
        return self._tables.get(table_name)

    def get_columns(self, table_name: str) -> List[ColumnInfo]:
        """Get all columns for a table."""
        self._ensure_loaded()
        table = self._tables.get(table_name)
        return table.columns if table else []

    def get_column(self, table_name: str, column_name: str) -> Optional[ColumnInfo]:
        """Get specific column information."""
        self._ensure_loaded()
        if table_name not in self._columns_by_table:
            return None
        return self._columns_by_table[table_name].get(column_name)

    def get_foreign_keys(self, table_name: str) -> List[ForeignKeyInfo]:
        """Get foreign key relationships for a table."""
        self._ensure_loaded()
        return self._fk_by_table.get(table_name, [])

    def get_company_column(self, table_name: str) -> Optional[str]:
        """Get the company column for a table based on its module."""
        self._ensure_loaded()
        table = self._tables.get(table_name)
        if not table:
            return None
        return self.MODULE_COMPANY_COLUMNS.get(table.module)

    def find_join_columns(self, table1: str, table2: str) -> List[Tuple[str, str]]:
        """
        Find columns that can be used to join two tables.

        Returns list of (table1_column, table2_column) pairs.
        """
        self._ensure_loaded()

        join_cols = []

        # Check foreign keys
        for fk in self._foreign_keys:
            if (fk.parent_table == table1 and fk.child_table == table2) or \
               (fk.parent_table == table2 and fk.child_table == table1):
                for p_col, c_col in zip(fk.parent_columns, fk.child_columns):
                    if fk.parent_table == table1:
                        join_cols.append((p_col, c_col))
                    else:
                        join_cols.append((c_col, p_col))

        return join_cols

    def get_all_table_names(self) -> Set[str]:
        """Get set of all known table/view names."""
        self._ensure_loaded()
        return set(self._tables.keys())

    def suggest_similar_tables(self, invalid_name: str, max_suggestions: int = 5) -> List[str]:
        """
        Suggest similar table names for an invalid name.

        Uses simple prefix matching for suggestions.
        """
        self._ensure_loaded()

        suggestions = []
        invalid_upper = invalid_name.upper()

        for table_name in self._tables.keys():
            # Check prefix match
            if table_name.upper().startswith(invalid_upper[:2]):
                suggestions.append(table_name)
            # Check if invalid name is substring
            elif invalid_upper in table_name.upper():
                suggestions.append(table_name)

        return sorted(suggestions)[:max_suggestions]

    def get_stats(self) -> Dict:
        """Get schema statistics."""
        self._ensure_loaded()

        tables = [t for t in self._tables.values() if t.object_type == 'TABLE']
        views = [t for t in self._tables.values() if t.object_type == 'VIEW']

        return {
            'total_objects': len(self._tables),
            'tables': len(tables),
            'views': len(views),
            'total_columns': sum(len(t.columns) for t in self._tables.values()),
            'foreign_keys': len(self._foreign_keys),
            'modules': list(set(t.module for t in self._tables.values()))
        }


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Schema Loader Utility")
    parser.add_argument('--vgpt2', type=str, default='C:/Github/VGPT2',
                        help='Path to VGPT2 repository')
    parser.add_argument('--stats', action='store_true', help='Show schema statistics')
    parser.add_argument('--check-table', type=str, help='Check if table exists')
    parser.add_argument('--check-column', type=str, nargs=2,
                        metavar=('TABLE', 'COLUMN'), help='Check if column exists')

    args = parser.parse_args()

    schema = SchemaLoader(args.vgpt2)

    if args.stats:
        stats = schema.get_stats()
        print("\nViewpoint Vista Schema Statistics")
        print("=" * 40)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    if args.check_table:
        exists = schema.table_exists(args.check_table)
        print(f"\nTable '{args.check_table}': {'EXISTS' if exists else 'NOT FOUND'}")
        if not exists:
            suggestions = schema.suggest_similar_tables(args.check_table)
            if suggestions:
                print(f"  Did you mean: {', '.join(suggestions)}")

    if args.check_column:
        table, column = args.check_column
        exists = schema.column_exists(table, column)
        print(f"\nColumn '{table}.{column}': {'EXISTS' if exists else 'NOT FOUND'}")
