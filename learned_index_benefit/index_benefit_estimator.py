import re
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
import pymysql
import json
import statistics
from functools import reduce
from math import log2
import os
import math
import sys
from datetime import datetime

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

class QueryCostEstimator:
    def __init__(self):
        self.conn = None
        self.table_sizes = {}
        self.column_stats = {}
        self.current_conditions = []
        
        # Configuration parameters for cost estimation
        self.config = {
            'page_size': 16384,  # Default InnoDB page size in bytes
            'key_overhead': 12,   # Bytes of overhead per key in index
            'cpu_cost_per_row': 0.2,  # CPU cost factor per row processed
            'io_cost_per_page': 1.0,  # I/O cost factor per page read
            'buffer_pool_size': 128 * 1024 * 1024,  # 128MB buffer pool size
            'min_selectivity': 0.0001,  # Minimum selectivity estimate
            'max_selectivity': 1.0      # Maximum selectivity estimate
        }
        
        self.connect_db()
        self._load_statistics()
        
    def connect_db(self):
        """Connect to PolarDB-IMCI database"""
        try:
            max_retries = 3
            retry_delay = 1
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    print(f"Attempting database connection (attempt {attempt + 1}/{max_retries})...")
                    self.conn = pymysql.connect(
                        host='172.17.0.1',
                        port=22224,
                        user='user1',
                        password='your_password',
                        database='tpch_sf1',
                        charset='utf8mb4',
                        cursorclass=pymysql.cursors.DictCursor
                    )
                    print("Successfully connected to PolarDB-IMCI")
                    return
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"Connection attempt failed: {e}")
                        print(f"Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
            
            raise last_error if last_error else Exception("Failed to connect to database")
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise
            
    def _load_statistics(self):
        cache_file = 'statistics.json'
        if os.path.exists(cache_file):
            print(f"Loading cached statistics from {cache_file}")
            with open(cache_file, 'r') as f:
                data=json.load(f)
            self.table_sizes = data['table_sizes']
            self.column_stats = {}
            for table, columns in data['column_stats'].items():
                self.column_stats[table] = {}
                for col_str, stats in columns.items():
                    column = Column(table, col_str)
                    self.column_stats[table][column] = stats
            print("Successfully loaded cached statistics")
        else:
            self.load_statistics()

            # Prepare data for caching
            cache_data = {
                'table_sizes': self.table_sizes,
                'column_stats': {
                    table: {str(col): stats for col, stats in cols.items()}
                    for table, cols in self.column_stats.items()
                }
            }
            
            # Save to cache file
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"Saved generated statistics to {cache_file} for future use")


    def load_statistics(self):
        """Load table and column statistics from the database"""
        if not self.conn:
            raise Exception("No database connection available")
            
        cursor = self.conn.cursor()
        try:
            print("Loading database statistics...")
            
            # Get table sizes and other metadata
            cursor.execute("""
                SELECT 
                    TABLE_NAME,
                    TABLE_ROWS,
                    AVG_ROW_LENGTH,
                    DATA_LENGTH,
                    INDEX_LENGTH
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = DATABASE()
            """)
            
            print("\nTable Statistics:")
            print(f"{'Table':<20} {'Rows':<12} {'Avg Row (B)':<12} {'Data (MB)':<12} {'Index (MB)':<12}")
            print("-" * 70)
            
            for row in cursor.fetchall():
                table_name = row['TABLE_NAME']
                rows = row['TABLE_ROWS']
                avg_row_len = row['AVG_ROW_LENGTH']
                data_len = row['DATA_LENGTH']
                idx_len = row['INDEX_LENGTH']
                table_name = table_name.lower()
                self.table_sizes[table_name] = rows if rows else 0
                
                # Convert lengths to MB for display
                data_mb = data_len / (1024 * 1024) if data_len else 0
                idx_mb = idx_len / (1024 * 1024) if idx_len else 0
                
                print(f"{table_name:<20} {rows or 0:<12} {avg_row_len or 0:<12} "
                      f"{data_mb:>9.2f} MB {idx_mb:>9.2f} MB")
            
            print("\nCollecting column statistics...")
            for table in self.table_sizes.keys():
                self.column_stats[table] = {}
                
                # First, analyze the table to update statistics
                print(f"\nAnalyzing table {table}...")
                cursor.execute(f"ANALYZE TABLE {table}")
                
                # Get column information
                cursor.execute(f"SHOW COLUMNS FROM {table}")
                columns = cursor.fetchall()
                
                print(f"\nColumn statistics for {table}:")
                print(f"{'Column':<20} {'Type':<15} {'Distinct':<10} {'Selectivity':<12}")
                print("-" * 60)
                
                for col in columns:
                    col_name = col['Field'].lower()
                    col_type = col['Type'].lower()
                    
                    # Get basic column statistics
                    try:
                        cursor.execute(f"""
                            SELECT 
                                COUNT(DISTINCT `{col_name}`) as distinct_count,
                                COUNT(*) as total_count
                            FROM {table}
                        """)
                        stats = cursor.fetchone()
                        
                        if stats:
                            distinct_count = int(stats['distinct_count']) if stats['distinct_count'] is not None else 0
                            total_count = int(stats['total_count']) if stats['total_count'] is not None else 0
                            selectivity = distinct_count / total_count if total_count > 0 else 1
                            
                            self.column_stats[table][col_name] = {
                                'type': col_type,
                                'distinct_count': distinct_count,
                                'total_count': total_count,
                                'avg_selectivity': selectivity
                            }
                            
                            print(f"{col_name:<20} {col_type:<15} {distinct_count:<10} "
                                  f"{selectivity:>10.4f}")
                            
                            # Get additional statistics for numeric columns
                            if any(numeric_type in col_type 
                                  for numeric_type in ('int', 'decimal', 'float', 'double')):
                                cursor.execute(f"""
                                    SELECT 
                                        MIN(`{col_name}`) as min_val,
                                        MAX(`{col_name}`) as max_val,
                                        AVG(`{col_name}`) as avg_val
                                    FROM {table}
                                """)
                                result = cursor.fetchone()
                                min_val = float(result['min_val']) if result['min_val'] is not None else 0
                                max_val = float(result['max_val']) if result['max_val'] is not None else 0
                                avg_val = float(result['avg_val']) if result['avg_val'] is not None else 0
                                
                                self.column_stats[table][col_name].update({
                                    'min_value': min_val,
                                    'max_value': max_val,
                                    'avg_value': avg_val
                                })
                                
                                print(f"  → Range: {min_val} to {max_val}, Avg: {avg_val:.2f}")
                                
                    except Exception as e:
                        print(f"Error collecting statistics for {table}.{col_name}: {e}")
            
            print("\nStatistics collection completed.")
        except Exception as e:
            print(f"Error loading statistics: {e}")
            raise
        finally:
            cursor.close()

    def convert_sql_to_mysql(self, sql: str) -> str:
        """Convert SQL with double quotes to MySQL format with backticks"""
        # Handle table.column patterns
        sql = re.sub(r'"([^"]+)"[.]"([^"]+)"', r'`\1`.`\2`', sql)
        # Handle remaining quoted identifiers
        sql = re.sub(r'"([^"]+)"', r'`\1`', sql)
        return sql

    def get_actual_query_cost(self, sql: str) -> float:
        """Get actual query cost from EXPLAIN output"""
        if not self.conn:
            return float('inf')
            
        # Convert SQL to MySQL format and clean up
        try:
            mysql_sql = self.convert_sql_to_mysql(sql)
            # Remove any leading/trailing whitespace and ensure proper spacing
            mysql_sql = ' '.join(mysql_sql.split())
            # Ensure the query starts with SELECT
            if not mysql_sql.upper().startswith('SELECT'):
                mysql_sql = 'SELECT ' + mysql_sql
            
            cursor = self.conn.cursor()
            try:
                # First try EXPLAIN FORMAT=JSON
                explain_sql = f"EXPLAIN FORMAT=JSON {mysql_sql}"
                # print(f"Executing: {explain_sql}")
                cursor.execute(explain_sql)
                plan_row = cursor.fetchone()
                
                if not plan_row:
                    print("No EXPLAIN plan returned")
                    return float('inf')
                if 'EXPLAIN' not in plan_row:
                    print("No EXPLAIN field in plan row")
                    return float('inf')
                if not plan_row['EXPLAIN']:
                    print("Empty EXPLAIN plan")
                    return float('inf')
                    
                try:
                    plan_data = json.loads(plan_row['EXPLAIN'])
                except json.JSONDecodeError as e:
                    print(f"Failed to parse EXPLAIN plan: {e}")
                    print(f"Raw plan: {plan_row['EXPLAIN']}")
                    return float('inf')
                    
                def extract_costs(node):
                    """Recursively extract costs from query plan"""
                    total_cost = 0
                    try:
                        if isinstance(node, dict):
                            # Get costs from current node
                            if 'rows_examined_per_scan' in node:
                                total_cost += float(node['rows_examined_per_scan'])
                            if 'rows' in node:
                                total_cost += float(node['rows'])
                            if 'cost_info' in node:
                                cost_info = node['cost_info']
                                if isinstance(cost_info, dict):
                                    if 'read_cost' in cost_info:
                                        total_cost += float(cost_info['read_cost'])
                                    if 'eval_cost' in cost_info:
                                        total_cost += float(cost_info['eval_cost'])
                            
                            # Process nested nodes
                            for key in ['nested_loop', 'materialized', 'table', 'subqueries', 'union_result']:
                                if key in node:
                                    value = node[key]
                                    if isinstance(value, list):
                                        for child in value:
                                            total_cost += extract_costs(child)
                                    elif isinstance(value, dict):
                                        total_cost += extract_costs(value)
                            
                            # Process attached_subqueries if present
                            if 'attached_subqueries' in node:
                                for subquery in node['attached_subqueries']:
                                    total_cost += extract_costs(subquery)
                            
                            # Process materialized_from_subquery if present
                            if 'materialized_from_subquery' in node:
                                total_cost += extract_costs(node['materialized_from_subquery'])
                            
                        elif isinstance(node, list):
                            for item in node:
                                total_cost += extract_costs(item)
                    except Exception as e:
                        print(f"Error extracting costs from node: {e}")
                        print(f"Node type: {type(node)}")
                        print(f"Node content: {node}")
                    return total_cost

                # Extract costs from JSON plan
                try:
                    if isinstance(plan_data, dict) and 'query_block' in plan_data:
                        return extract_costs(plan_data['query_block'])
                    elif isinstance(plan_data, list):
                        total_cost = 0
                        for block in plan_data:
                            if isinstance(block, dict) and 'query_block' in block:
                                total_cost += extract_costs(block['query_block'])
                        return total_cost
                    else:
                        print(f"Unexpected plan data format: {type(plan_data)}")
                        return float('inf')
                except Exception as e:
                    # If JSON format fails, try traditional EXPLAIN
                    cursor.execute(f"EXPLAIN {mysql_sql}")
                    plan = cursor.fetchall()
                    if not plan:
                        print(f"No EXPLAIN plan returned for query: {mysql_sql[:100]}...")
                        return float('inf')
                    
                    # Calculate cost from traditional EXPLAIN output
                    total_cost = 0
                    for row in plan:
                        # Use rows examined as cost approximation
                        rows = int(row.get('rows', 0))
                        total_cost += rows
                    return float(total_cost)
                    
            except Exception as e:
                print(f"Error executing EXPLAIN: {e}")
                print(f"Query: {mysql_sql[:100]}...")
                return float('inf')
            finally:
                cursor.close()
                
        except Exception as e:
            print(f"Error converting SQL: {e}")
            print(f"Original query: {sql[:100]}...")
            return float('inf')
            
        return float('inf')  # Default return if all attempts fail

    def estimate_scan_cost(self, table: str) -> float:
        """Estimate cost of full table scan with improved scaling"""
        if not self.conn or table not in self.table_sizes:
            return float('inf')
            
        table_size = self.table_sizes[table]
        if table_size == 0:
            return 1.0
            
        # Get actual table statistics
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"""
                SELECT 
                    AVG_ROW_LENGTH as avg_row_length,
                    DATA_LENGTH as data_length,
                    INDEX_LENGTH as index_length
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = '{table}'
            """)
            table_status = cursor.fetchone()
            
            page_size = 16384  # 16KB InnoDB page size
            if table_status and table_status['avg_row_length']:
                avg_row_length = int(table_status['avg_row_length'])
                data_length = int(table_status['data_length'])
                data_pages = (data_length + page_size - 1) // page_size
            else:
                avg_row_length = 100  # Default assumption
                data_pages = table_size // (page_size // avg_row_length)
            
            # Base I/O cost calculation
            base_io_cost = data_pages * 0.05  # Sequential read is efficient
            
            # CPU cost with progressive scaling
            if table_size < 1000:
                cpu_factor = 0.001  # Very small tables
            elif table_size < 10000:
                cpu_factor = 0.005  # Small tables
            elif table_size < 100000:
                cpu_factor = 0.01   # Medium tables
            else:
                # Progressive scaling for large tables
                cpu_factor = 0.01 * (1 + log2(table_size / 100000) * 0.1)
            
            cpu_cost = table_size * cpu_factor
            
            # Memory buffer consideration
            buffer_size = 1024 * 1024 * 1024  # Assume 1GB buffer pool
            buffer_hit_ratio = min(0.9, buffer_size / (data_pages * page_size))
            io_cost = base_io_cost * (1 - buffer_hit_ratio)
            
            # Total cost with progressive scaling
            total_cost = io_cost + cpu_cost
            
            # # Apply correction factors based on observed patterns
            # if total_cost > 50000:  # Very high estimates
            #     total_cost *= 0.1   # Reduce overestimation
            # elif total_cost > 10000:  # High estimates
            #     total_cost *= 0.3   # Moderate reduction
            # elif total_cost < 100:  # Low estimates
            #     total_cost *= 2.0   # Increase underestimation
                
            return max(total_cost, 1.0)  # Ensure non-zero cost
            
        except Exception as e:
            print(f"Error getting table statistics: {e}")
            return max(self.table_sizes.get(table, 1000000), 1.0)
            
        finally:
            cursor.close()

    def _get_column_size(self, column_name: str) -> int:
        """Estimate the average size of a column in bytes"""
        # Default sizes based on common data types
        if 'int' in column_name.lower():
            return 4  # Integer
        elif 'date' in column_name.lower():
            return 3  # Date
        elif 'char' in column_name.lower():
            return 20  # Variable character
        elif 'decimal' in column_name.lower():
            return 8  # Decimal
        return 12  # Default size

    def _get_column_selectivity(self, table: str, column: str, operator: str, value) -> float:
        """Calculate column selectivity based on operator and value"""
        if table not in self.column_stats or column not in self.column_stats[table]:
            return self.config['max_selectivity']
            
        stats = self.column_stats[table][column]
        distinct_count = stats['distinct_count']
        total_count = stats['total_count']
        
        if total_count == 0:
            return self.config['max_selectivity']
            
        base_selectivity = max(
            self.config['min_selectivity'],
            min(1.0 / distinct_count if distinct_count > 0 else 1.0, 
                self.config['max_selectivity'])
        )
        
        if operator == '=':
            return base_selectivity
        elif operator in ('>', '<', '>=', '<='):
            return min(0.3, base_selectivity * 3)  # Assume range queries are less selective
        elif operator == 'LIKE':
            if str(value).startswith('%') and str(value).endswith('%'):
                return min(0.5, base_selectivity * 5)  # Contains pattern
            elif str(value).startswith('%'):
                return min(0.3, base_selectivity * 3)  # Ends with pattern
            elif str(value).endswith('%'):
                return min(0.2, base_selectivity * 2)  # Starts with pattern
            return base_selectivity  # Exact pattern
        elif operator == 'IN':
            try:
                in_list_size = len(value) if isinstance(value, (list, tuple)) else 1
                return min(base_selectivity * in_list_size, self.config['max_selectivity'])
            except:
                return min(0.1, base_selectivity * 2)
        return self.config['max_selectivity']

    def _combine_selectivities(self, selectivities: list, index: Index) -> float:
        """Combine multiple selectivities for a compound index"""
        if not selectivities:
            return 1.0
            
        # For unique indexes, use the minimum selectivity
        if index and index.is_unique:
            return min(selectivities)
            
        # For non-unique indexes, use a combination approach
        combined = reduce(lambda x, y: x * y, selectivities)
        return max(combined, self.config['min_selectivity'])

    def _get_clustering_factor(self, table: str, index: Optional[Index]) -> float:
        """Estimate clustering factor for an index"""
        if not index:
            return 1.0
            
        # For primary key or unique indexes, assume good clustering
        if index.is_unique:
            return 0.2
            
        # For foreign key indexes, assume moderate clustering
        if any('_fk' in str(col) for col in index.columns):
            return 0.5
            
        # For other indexes, assume poor clustering
        return 0.8

    def _calculate_io_cost(self, table: str, rows: int, index: Optional[Index],
                          operator: Optional[str], clustering_factor: float) -> float:
        """Calculate I/O cost for accessing rows"""
        if table not in self.table_sizes:
            return float('inf')
            
        total_rows = self.table_sizes[table]
        if total_rows == 0:
            return 1.0
            
        # Calculate pages needed for table scan
        avg_row_size = 100  # Assume average row size of 100 bytes
        rows_per_page = self.config['page_size'] / avg_row_size
        total_pages = math.ceil(total_rows / rows_per_page)
        
        if not index:
            # Full table scan
            return total_pages * self.config['io_cost_per_page']
            
        # Index scan
        index_pages = math.ceil(math.log(total_rows, 2))  # B-tree height
        data_pages = math.ceil(rows * clustering_factor)
        
        return (index_pages + data_pages) * self.config['io_cost_per_page']

    def _calculate_buffer_hit_ratio(self, table: str, rows: int) -> float:
        """Estimate buffer pool hit ratio"""
        if table not in self.table_sizes:
            return 0.0
            
        total_rows = self.table_sizes[table]
        if total_rows == 0:
            return 1.0
            
        # Estimate working set size
        avg_row_size = 100  # Assume average row size of 100 bytes
        working_set_size = rows * avg_row_size
        
        # Calculate hit ratio based on buffer pool size
        hit_ratio = min(1.0, self.config['buffer_pool_size'] / working_set_size)
        return max(0.1, hit_ratio)  # Minimum 10% hit ratio

    def estimate_index_lookup_cost(self, table: str, selectivity: float, 
                                 index: Optional[Index] = None, 
                                 operator: str = None) -> float:
        """Estimate cost of index lookup with enhanced modeling"""
        if table not in self.table_sizes:
            return float('inf')

        total_rows = self.table_sizes[table]
        if total_rows == 0:
            return 1.0  # Minimum cost for empty table

        # Calculate B-tree parameters
        avg_key_size = sum(
            self._get_column_size(col.name) 
            for col in (index.columns if index else [])
        ) / max(1, len(index.columns)) if index else 20
        
        keys_per_page = self.config['page_size'] // (avg_key_size + self.config['key_overhead'])
        fanout = keys_per_page * 0.7  # Average fill factor
        btree_height = 1 if total_rows == 0 else math.ceil(math.log(total_rows, fanout))

        # Calculate selectivity using actual statistics
        if index and hasattr(self, '_current_conditions') and self._current_conditions:
            actual_selectivities = []
            for col in index.columns:
                for cond in self._current_conditions:
                    if cond[0] == table and cond[1] == col.name:
                        actual_selectivities.append(
                            self._get_column_selectivity(table, col.name, cond[2], cond[3])
                        )
            if actual_selectivities:
                selectivity = self._combine_selectivities(actual_selectivities, index)

        # Calculate base cost with more conservative scaling
        # First, determine access pattern and rows
        if index:
            if index.is_unique and operator == '=':
                # Point lookup on unique index
                base_rows = 1
                access_factor = 1.0
            elif operator == '=':
                # Point lookup on non-unique index
                base_rows = min(int(total_rows * selectivity), 10)
                access_factor = 1.2
            elif operator in ('>', '<', '>=', '<='):
                # Range scan
                base_rows = min(int(total_rows * selectivity), int(total_rows * 0.01))
                access_factor = 1.5
            elif operator == 'LIKE':
                if any(cond[3].strip("'").startswith('%') and cond[3].strip("'").endswith('%') 
                      for cond in self._current_conditions if cond[0] == table):
                    # Contains pattern
                    base_rows = min(int(total_rows * selectivity), int(total_rows * 0.02))
                    access_factor = 2.0
                else:
                    # Starts/ends with pattern
                    base_rows = min(int(total_rows * selectivity), int(total_rows * 0.01))
                    access_factor = 1.5
            else:
                # Other index operations
                base_rows = min(int(total_rows * selectivity), int(total_rows * 0.05))
                access_factor = 1.8
        else:
            # Full table scan
            base_rows = min(int(total_rows * selectivity), int(total_rows * 0.1))
            access_factor = 2.0

        # Calculate base cost using logarithmic scaling
        if base_rows == 1:
            # Point lookup
            base_cost = 1.0
        elif base_rows <= 100:
            # Small result set
            base_cost = math.log2(base_rows + 1) * access_factor
        elif base_rows <= 10000:
            # Medium result set
            base_cost = math.sqrt(base_rows) * access_factor
        else:
            # Large result set
            base_cost = math.pow(base_rows, 1/3) * access_factor

        # Apply buffer pool impact
        buffer_hit_ratio = min(0.9, self._calculate_buffer_hit_ratio(table, base_rows) * 1.2)
        total_cost = base_cost * (1 - buffer_hit_ratio)

        # Add minimal CPU cost
        cpu_cost = math.log2(base_rows + 1) * 0.1

        # Final cost with bounds
        total_cost = min(
            total_cost + cpu_cost,
            math.sqrt(total_rows),  # Sublinear scaling with table size
            100.0  # Hard cap at 100
        )
        
        return max(1.0, total_cost)

    def validate_cost_model(self, query: str, indexes: List[Index] = None) -> Dict:
        """Validate the cost model by comparing estimated vs actual costs"""
        # Get actual cost from database
        actual_cost = self.get_actual_query_cost(query)
        
        # Parse query to get tables and conditions
        try:
            # Extract conditions from query for better selectivity estimation
            self._current_conditions = []  # Reset conditions
            tables = set()
            
            # Simple query parsing (can be enhanced)
            query_lower = query.lower()
            from_idx = query_lower.find('from')
            where_idx = query_lower.find('where')
            
            if from_idx > 0:
                # Extract table names
                tables_part = query_lower[from_idx:where_idx if where_idx > 0 else None]
                tables = {t.strip() for t in tables_part.replace('from', '').split(',') if t.strip()}
                
                # Extract conditions if WHERE clause exists
                if where_idx > 0:
                    where_part = query_lower[where_idx:]
                    # Simple condition parsing (can be enhanced)
                    conditions = where_part.replace('where', '').split('and')
                    for cond in conditions:
                        parts = cond.strip().split()
                        if len(parts) >= 3:
                            col_parts = parts[0].split('.')
                            if len(col_parts) == 2:
                                table, column = col_parts
                                operator = parts[1]
                                value = ' '.join(parts[2:])
                                self._current_conditions.append((table, column, operator, value))
            
            # Estimate cost for each table and index combination
            total_estimated_cost = 0
            for table in tables:
                table_indexes = [idx for idx in (indexes or []) if idx.table == table]
                if table_indexes:
                    # Use the best index for each table
                    costs = [self.estimate_index_lookup_cost(table, 0.1, idx) for idx in table_indexes]
                    total_estimated_cost += min(costs)
                else:
                    # No index available, use table scan cost
                    total_estimated_cost += self.estimate_index_lookup_cost(table, 1.0)
            
            # Calculate error metrics
            error_ratio = abs(total_estimated_cost - actual_cost) / max(1.0, actual_cost)
            is_accurate = error_ratio <= 0.5  # Consider estimate accurate if within 50% of actual
            
            return {
                'estimated_cost': total_estimated_cost,
                'actual_cost': actual_cost,
                'error_ratio': error_ratio,
                'is_accurate': is_accurate
            }
            
        except Exception as e:
            print(f"Error validating cost model: {e}")
            return {
                'estimated_cost': float('inf'),
                'actual_cost': actual_cost,
                'error_ratio': float('inf'),
                'is_accurate': False
            }

    def _get_column_size(self, col_name: str) -> int:
        """Estimate the average size of a column in bytes based on column name and statistics"""
        # Check if we have actual statistics for this column
        for table, cols in self.column_stats.items():
            if col_name in cols:
                col_stats = cols[col_name]
                if 'type' in col_stats:
                    col_type = col_stats['type'].lower()
                    # Integer types
                    if 'tinyint' in col_type:
                        return 1
                    elif 'smallint' in col_type:
                        return 2
                    elif 'mediumint' in col_type:
                        return 3
                    elif 'int' in col_type:
                        return 4
                    elif 'bigint' in col_type:
                        return 8
                    # Floating point types
                    elif 'float' in col_type:
                        return 4
                    elif 'double' in col_type:
                        return 8
                    elif 'decimal' in col_type:
                        # Extract precision and scale if available
                        match = re.search(r'decimal\((\d+),(\d+)\)', col_type)
                        if match:
                            precision, scale = map(int, match.groups())
                            return (precision + 2) // 2  # Approximate bytes needed
                        return 8
                    # Date and time types
                    elif 'date' in col_type:
                        return 3
                    elif 'time' in col_type:
                        return 3
                    elif 'datetime' in col_type or 'timestamp' in col_type:
                        return 8
                    # String types
                    elif 'varchar' in col_type or 'char' in col_type:
                        # Extract length if available
                        match = re.search(r'\((\d+)\)', col_type)
                        if match:
                            length = int(match.group(1))
                            return min(length, 255)  # Cap at 255 bytes
                        return 32  # Default assumption for varchar
                    elif 'text' in col_type:
                        return 64  # Average size assumption for text
                    elif 'blob' in col_type:
                        return 128  # Average size assumption for blob
        
        # Fallback based on column name hints
        col_name_lower = col_name.lower()
        if 'id' in col_name_lower or 'key' in col_name_lower:
            return 4  # Assume it's an integer key
        elif 'date' in col_name_lower or 'time' in col_name_lower:
            return 8  # Assume it's a datetime
        elif 'name' in col_name_lower or 'desc' in col_name_lower:
            return 32  # Assume it's a string
        elif 'num' in col_name_lower or 'count' in col_name_lower:
            return 4  # Assume it's a number
        
        return 16  # Default assumption

    def _combine_selectivities(self, selectivities: List[float], index: Index) -> float:
        """Combine selectivities for composite indexes with correlation awareness"""
        if not selectivities:
            return 1.0
            
        # For unique indexes, use the minimum selectivity
        if index and index.is_unique:
            return min(selectivities)
            
        # For non-unique indexes, use a correlation-aware combination
        combined = selectivities[0]
        for i, sel in enumerate(selectivities[1:], 1):
            # Assume decreasing correlation for subsequent columns
            correlation = 0.8 ** i
            # Combine selectivities with correlation factor
            combined *= sel ** correlation
        return min(1.0, combined)

    def _get_clustering_factor(self, table: str, index: Optional[Index]) -> float:
        """Estimate clustering factor for an index"""
        if not index:
            return 1.0
            
        # For primary key or unique indexes, assume good clustering
        if index.is_unique:
            return 0.2
            
        # For foreign key indexes, assume moderate clustering
        if any('_fk' in str(col) for col in index.columns):
            return 0.5
            
        # For other indexes, assume poor clustering
        return 0.8

    def _calculate_io_cost(self, table: str, rows: int, index: Optional[Index],
                          operator: Optional[str], clustering_factor: float) -> float:
        """Calculate I/O cost for accessing rows"""
        if table not in self.table_sizes:
            return float('inf')
            
        total_rows = self.table_sizes[table]
        if total_rows == 0:
            return 1.0
            
        # Calculate pages needed for table scan
        avg_row_size = 100  # Assume average row size of 100 bytes
        rows_per_page = self.config['page_size'] / avg_row_size
        total_pages = math.ceil(total_rows / rows_per_page)
        
        if not index:
            # Full table scan
            return total_pages * self.config['io_cost_per_page']
            
        # Index scan
        index_pages = math.ceil(math.log(total_rows, 2))  # B-tree height
        data_pages = math.ceil(rows * clustering_factor)
        
        # Adjust cost based on operator type
        if operator in ('=', 'IN'):
            # Point queries are more efficient
            return (index_pages + data_pages) * self.config['io_cost_per_page'] * 0.8
        elif operator in ('>', '<', '>=', '<='):
            # Range queries require more I/O
            return (index_pages + data_pages) * self.config['io_cost_per_page'] * 1.2
        else:
            return (index_pages + data_pages) * self.config['io_cost_per_page']

    def _calculate_buffer_hit_ratio(self, table: str, rows: int) -> float:
        """Calculate buffer pool hit ratio"""
        if table not in self.table_sizes:
            return 0.0
            
        total_rows = self.table_sizes[table]
        if total_rows == 0:
            return 1.0
            
        # Estimate working set size
        avg_row_size = 100  # Assume average row size of 100 bytes
        working_set_size = rows * avg_row_size
        
        # Calculate hit ratio based on buffer pool size and access pattern
        hit_ratio = min(1.0, self.config['buffer_pool_size'] / working_set_size)
        
        # Adjust for locality of reference
        if rows < 1000:
            # Small result sets have better locality
            hit_ratio = min(1.0, hit_ratio * 1.5)
        elif rows > total_rows * 0.5:
            # Large scans have worse locality
            hit_ratio = hit_ratio * 0.5
            
        return max(0.1, min(0.99, hit_ratio))  # Keep between 10% and 99%

    def _extract_joins(self, sql: str) -> List[tuple]:
        """Extract join conditions from SQL"""
        joins = []
        # Match JOIN...ON patterns
        join_pattern = r'JOIN\s+`([^`]+)`\s+ON\s+`([^`]+)`\.`([^`]+)`\s*=\s*`([^`]+)`\.`([^`]+)`'
        for match in re.finditer(join_pattern, sql.lower(), re.IGNORECASE):
            joined_table, t1, c1, t2, c2 = match.groups()
            joins.append((t1, c1, t2, c2))
        return joins

    def _estimate_join_selectivity(self, t1: str, c1: str, t2: str, c2: str) -> float:
        """Estimate join selectivity between two tables"""
        # Get column statistics
        t1_stats = self.column_stats.get(t1, {}).get(c1, {})
        t2_stats = self.column_stats.get(t2, {}).get(c2, {})
        
        if not t1_stats or not t2_stats:
            return 0.01  # Conservative default
            
        # Get distinct values and total rows
        t1_distinct = t1_stats.get('distinct_count', 1)
        t2_distinct = t2_stats.get('distinct_count', 1)
        t1_total = t1_stats.get('total_count', 1)
        t2_total = t2_stats.get('total_count', 1)
        
        if t1_distinct == 0 or t2_distinct == 0:
            return 0.1  # More conservative than 1.0
            
        # Calculate average number of duplicates
        t1_dups = t1_total / max(1, t1_distinct)
        t2_dups = t2_total / max(1, t2_distinct)
        
        # Detect join type
        is_pk = t1_dups <= 1.1 or t2_dups <= 1.1  # Primary key
        is_fk = (t1_dups > 1.1 and t1_dups <= 2) or (t2_dups > 1.1 and t2_dups <= 2)  # Foreign key
        is_many = t1_dups > 10 or t2_dups > 10  # Many-to-many
        
        # Calculate base selectivity
        if is_pk and (is_fk or is_many):
            # Primary-Foreign key join
            selectivity = 1.0 / max(t1_distinct, t2_distinct)
            return min(0.01, selectivity)  # Cap at 1%
        elif is_pk and is_pk:
            # Primary-Primary key join (unlikely but possible)
            return 0.001  # Very selective
        elif is_many:
            # Many-to-many join
            avg_rows = math.sqrt(t1_total * t2_total)
            selectivity = math.log2(avg_rows) / avg_rows
            return min(0.1, selectivity)  # Cap at 10%
        else:
            # Regular join
            selectivity = 1.0 / math.pow(max(t1_distinct, t2_distinct), 1/3)  # Cube root for less aggressive reduction
            return min(0.05, selectivity)  # Cap at 5%

    def estimate_query_cost(self, sql: str, available_indexes: List[Index] = None) -> float:
        """Estimate the cost of executing a query with given indexes"""
        try:
            # Handle quoted identifiers
            sql = self.convert_sql_to_mysql(sql)
            query_lower = sql.lower()
            
            # Extract tables, conditions, and joins
            tables = set()
            conditions = []
            joins = self._extract_joins(query_lower)
            
            # Extract tables from FROM and JOIN clauses
            from_idx = query_lower.find('from')
            where_idx = query_lower.find('where')
            if from_idx > 0:
                # Extract table names
                tables_part = query_lower[from_idx:where_idx if where_idx > 0 else None]
                table_matches = re.findall(r'`([^`]+)`', tables_part)
                tables = {t.strip() for t in table_matches if t.strip()}
                
                # Extract conditions if WHERE clause exists
                if where_idx > 0:
                    where_part = query_lower[where_idx:]
                    conditions = re.findall(r'`([^`]+)`\.`([^`]+)`\s*([=<>!]+|like|in|not\s+like|not\s+in)\s*([^and]+)', where_part, re.IGNORECASE)
            
            # Calculate base cost for each table
            table_costs = {}
            for table in tables:
                # Get table size
                table_size = self.table_sizes.get(table, 1000000)
                
                # Find applicable indexes
                table_indexes = [idx for idx in (available_indexes or []) if idx.table == table]
                
                # Calculate selectivity based on conditions
                table_conditions = [c for c in conditions if c[0] == table]
                selectivity = 1.0
                
                if table_conditions:
                    # Calculate combined selectivity of all conditions
                    condition_selectivities = []
                    for _, col, op, val in table_conditions:
                        if col in self.column_stats.get(table, {}):
                            stats = self.column_stats[table][col]
                            if op == '=':
                                sel = 1.0 / stats['distinct_count'] if stats['distinct_count'] > 0 else 0.1
                            elif op in ('>', '<', '>=', '<='):
                                sel = 0.3  # Range queries
                            elif 'like' in op.lower():
                                if val.strip("'").startswith('%') and val.strip("'").endswith('%'):
                                    sel = 0.2  # Contains pattern
                                else:
                                    sel = 0.1  # Starts/ends with pattern
                            elif 'in' in op.lower():
                                # Count number of values in IN clause
                                val_count = len(re.findall(r"'[^']*'", val))
                                sel = min(0.9, val_count * (1.0 / stats['distinct_count'])) if stats['distinct_count'] > 0 else 0.1
                            else:
                                sel = 0.5
                            condition_selectivities.append(sel)
                    
                    # Combine selectivities (assume some independence)
                    selectivity = reduce(lambda x, y: x * math.sqrt(y), condition_selectivities, 1.0)
                
                # Calculate access cost
                if table_indexes:
                    # Find best index
                    min_cost = float('inf')
                    for idx in table_indexes:
                        idx_cols = {col.name for col in idx.columns}
                        matching_conditions = [c for c in table_conditions if c[1] in idx_cols]
                        if matching_conditions:
                            cost = self.estimate_index_lookup_cost(table, selectivity, idx)
                            min_cost = min(min_cost, cost)
                    
                    if min_cost == float('inf'):
                        # No useful index found
                        min_cost = table_size * selectivity
                else:
                    # Table scan
                    min_cost = table_size * selectivity
                
                table_costs[table] = min_cost
            
            # Apply join costs
            if not joins:
                # No joins, just sum the individual table costs
                total_cost = sum(table_costs.values())
            else:
                # Calculate join cost using a more aggressive approach
                # Group tables by join relationships
                join_graph = {}
                for t1, c1, t2, c2 in joins:
                    if t1 not in join_graph:
                        join_graph[t1] = set()
                    if t2 not in join_graph:
                        join_graph[t2] = set()
                    join_graph[t1].add(t2)
                    join_graph[t2].add(t1)
                
                # Find connected components (join groups)
                def find_connected_tables(start, visited):
                    component = {start}
                    stack = [start]
                    while stack:
                        table = stack.pop()
                        for neighbor in join_graph.get(table, []):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                component.add(neighbor)
                                stack.append(neighbor)
                    return component
                
                visited = set()
                join_groups = []
                for table in table_costs:
                    if table not in visited and table in join_graph:
                        group = find_connected_tables(table, visited)
                        join_groups.append(group)
                
                # Calculate cost for each join group
                total_cost = 0
                for group in join_groups:
                    # Start with smallest table in group
                    group_tables = [(t, table_costs[t]) for t in group]
                    group_tables.sort(key=lambda x: x[1])
                    
                    # Calculate join cost with improved estimation
                    smallest_table = group_tables[0][1]
                    largest_table = group_tables[-1][1]
                    joined_tables = {group_tables[0][0]}
                    
                    # Find all join conditions for this group
                    group_joins = [
                        (t1, c1, t2, c2) for t1, c1, t2, c2 in joins
                        if t1 in group and t2 in group
                    ]
                    
                    # Calculate overall group selectivity
                    group_selectivity = 1.0
                    for t1, c1, t2, c2 in group_joins:
                        sel = self._estimate_join_selectivity(t1, c1, t2, c2)
                        # Use root based on group size for less aggressive reduction
                        group_selectivity *= math.pow(sel, 1/len(group))
                    
                    # Base join cost starts with smallest table
                    join_cost = smallest_table
                    
                    # Process remaining tables in optimal order
                    for table, cost in group_tables[1:]:
                        # Find direct join conditions
                        direct_joins = [
                            (t1, c1, t2, c2) for t1, c1, t2, c2 in group_joins
                            if ((t1 == table and t2 in joined_tables) or
                                (t2 == table and t1 in joined_tables))
                        ]
                        
                        # Calculate join factor based on conditions
                        if direct_joins:
                            # Use minimum selectivity of direct joins
                            direct_selectivity = min(
                                self._estimate_join_selectivity(t1, c1, t2, c2)
                                for t1, c1, t2, c2 in direct_joins
                            )
                            # Use logarithmic scaling for direct joins
                            join_factor = math.log2(cost + 1) * direct_selectivity
                        else:
                            # No direct joins - use more conservative estimate
                            join_factor = math.sqrt(cost) * group_selectivity
                        
                        # Apply cost scaling with improved bounds
                        join_cost = min(
                            join_cost * (1 + join_factor),  # Join with selectivity
                            join_cost + math.sqrt(cost),  # Sublinear increase
                            cost * 0.2  # Cap at 20% of table size
                        )
                        joined_tables.add(table)
                    
                    # Apply final adjustments to join cost
                    # Scale based on available indexes and join conditions
                    if any(idx for idx in (available_indexes or []) 
                          if idx.table in group and idx.is_unique):
                        # Group contains unique indexes - better join efficiency
                        join_cost *= 0.3  # 70% reduction
                    elif len(group_joins) >= len(group) - 1:
                        # All tables are directly joined - optimal case
                        join_cost *= 0.5  # 50% reduction
                    elif len(group_joins) > 0:
                        # Some tables are joined - partial optimization
                        join_cost *= 0.7  # 30% reduction
                    
                    # Apply group size scaling
                    group_size = len(group)
                    if group_size > 2:
                        # Use logarithmic scaling for multiple joins
                        scaling_factor = math.log2(group_size) * 0.2
                        join_cost *= (1 + scaling_factor)
                    
                    # Final bounds for group cost
                    group_cost = min(
                        join_cost,
                        smallest_table * math.sqrt(group_size),  # Scale with group size
                        largest_table * 0.1,  # Cap at 10% of largest table
                        sum(cost for _, cost in group_tables) * 0.05  # Cap at 5% of total
                    )
                    
                    total_cost += group_cost
                
                # Add costs for non-joined tables with minimal reduction
                non_joined_tables = set(table_costs.keys()) - set().union(*join_groups)
                for table in non_joined_tables:
                    total_cost += table_costs[table] * 0.8  # 20% reduction
                
                # Apply final scaling based on query complexity
                join_count = len(joins)
                if join_count > 1:
                    # Use logarithmic scaling with moderate reduction
                    scaling_factor = math.log2(join_count) * 0.1  # 90% reduction in join overhead
                    total_cost *= (1 + scaling_factor)
                
                # Apply final bounds with conservative capping
                max_table_size = max(table_costs.values())
                total_tables = len(table_costs)
                
                if total_tables == 1:
                    # Single table query - use table size as reference
                    total_cost = min(
                        total_cost,
                        max_table_size * 0.2  # Cap at 20% of table size
                    )
                elif total_tables == 2:
                    # Two-table join - use smaller table as reference
                    total_cost = min(
                        total_cost,
                        min(table_costs.values()) * math.sqrt(max_table_size)  # Scale with size difference
                    )
                else:
                    # Multi-table join - use more complex scaling
                    total_cost = min(
                        total_cost,
                        max_table_size * math.log2(total_tables),  # Scale with table count
                        sum(table_costs.values()) * 0.1  # Cap at 10% of total size
                    )
                
                # Final adjustment based on join complexity and conditions
                if join_count > 5:
                    # Complex joins with good conditions
                    if len([idx for idx in (available_indexes or []) if idx.is_unique]) >= join_count / 2:
                        total_cost *= 0.2  # 80% reduction - many unique indexes
                    else:
                        total_cost *= 0.5  # 50% reduction - fewer indexes
                elif join_count > 2:
                    # Medium complexity joins
                    total_cost *= 0.7  # 30% reduction
                
                total_cost = max(1.0, total_cost)  # Ensure minimum cost of 1.0
            
            return max(1.0, total_cost)
            
        except Exception as e:
            print(f"Error estimating query cost: {e}")
            return float('inf')

    def validate_cost_model(self, query: str, indexes: List[Index] = None) -> Dict:
        """Validate the cost model by comparing estimated vs actual costs"""
        # Get actual cost from database
        actual_cost = self.get_actual_query_cost(query)
        
        # Get estimated cost
        estimated_cost = self.estimate_query_cost(query, indexes)
        
        # Calculate error metrics
        if actual_cost > 0 and estimated_cost != float('inf'):
            error_ratio = abs(estimated_cost - actual_cost) / actual_cost
            is_accurate = 0.5 <= estimated_cost / actual_cost <= 2.0
        else:
            error_ratio = float('inf')
            is_accurate = False
        
        return {
            'estimated_cost': estimated_cost,
            'actual_cost': actual_cost,
            'error_ratio': error_ratio,
            'is_accurate': is_accurate
        }
