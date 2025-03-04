from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Column:
    table: str
    name: str
    
    def __str__(self):
        return f"{self.table}.{self.name}"
        
    def __hash__(self):
        return hash((self.table, self.name))

@dataclass(frozen=True)
class Index:
    table: str
    columns: tuple
    is_unique: bool = False
    
    def __init__(self, table: str, columns: List[Column], is_unique: bool = False):
        object.__setattr__(self, 'table', table)
        object.__setattr__(self, 'columns', tuple(columns))
        object.__setattr__(self, 'is_unique', is_unique)
    
    def __str__(self):
        return f"INDEX ON {self.table}({', '.join(str(c) for c in self.columns)})"
        
    def __hash__(self):
        return hash((self.table, self.columns, self.is_unique))

@dataclass
class TableStatistics:
    name: str
    row_count: int
    avg_row_length: int
    data_length: int
    index_length: int
    column_stats: dict  # Dict[column_name, ColumnStatistics]

@dataclass
class ColumnStatistics:
    name: str
    data_type: str
    distinct_count: int
    total_count: int
    avg_selectivity: float
    min_value: float = None
    max_value: float = None
    avg_value: float = None