import re
from typing import List, Dict, Set, Optional
from math import log2
from functools import reduce
from models.data_models import Index, Column, TableStatistics
from database.db_connector import DatabaseConnector
from utils.sql_converter import SQLConverter

class QueryCostEstimator:
    def __init__(self, db_connector: DatabaseConnector):
        self.db = db_connector
        self.table_stats: Dict[str, TableStatistics] = {}
        # self.load_statistics()

    def load_statistics(self):
        """Load database statistics"""
        self.table_stats = self.db.collect_statistics()

    def parse_query(self, sql: str) -> tuple[Set[str], List[str]]:
        """Parse SQL query to extract tables and conditions"""
        tables = set()
        conditions = []
        
        # Extract table names from FROM clause
        from_match = re.search(r'FROM\s+([^WHERE|GROUP|ORDER|LIMIT|;]+)', sql, re.IGNORECASE)
        if from_match:
            tables_str = from_match.group(1)
            # Handle table aliases
            table_refs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?)', tables_str)
            for ref in table_refs:
                table = ref.split()[0].lower()  # Get the actual table name
                tables.add(table)

        # Extract conditions from WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|;|$)', sql, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            # Split conditions on AND
            conditions = [cond.strip() for cond in where_clause.split('AND')]

        return tables, conditions

    def estimate_base_cost(self, tables: Set[str]) -> float:
        """Estimate base cost without considering indexes"""
        base_cost = 0
        for table in tables:
            if table in self.table_stats:
                stats = self.table_stats[table]
                # Calculate I/O cost based on data size and row length
                io_cost = (stats.data_length / 16384)  # Assuming 16KB page size
                # Calculate CPU cost based on number of rows
                cpu_cost = stats.row_count * 0.1  # CPU cost per row
                # Add memory access cost
                mem_cost = stats.row_count * stats.avg_row_length / 1024  # Memory cost based on data size
                
                # Combine costs with appropriate weights
                table_cost = (io_cost * 2.0 +  # I/O is typically most expensive
                            cpu_cost * 1.0 +   # CPU cost
                            mem_cost * 0.5)    # Memory access cost
                
                base_cost += table_cost
        
        return max(100.0, base_cost)  # Ensure minimum cost reflects basic overhead

    def estimate_condition_selectivity(self, table: str, condition: str) -> float:
        """Estimate selectivity of a condition"""
        # Extract column name from condition
        col_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*([=<>!]+|LIKE|IN|NOT IN)', condition)
        if not col_match:
            return 1.0

        col_name = col_match.group(1)
        operator = col_match.group(2)

        if table not in self.table_stats or col_name not in self.table_stats[table].column_stats:
            return 1.0

        col_stats = self.table_stats[table].column_stats[col_name]
        
        # Estimate selectivity based on operator and column statistics
        if operator == '=':
            return 1.0 / max(1, col_stats.distinct_count)
        elif operator in ('>', '<', '>=', '<='):
            return 0.3  # Assume range conditions select about 30% of data
        elif operator == 'LIKE':
            if '%' in condition:
                return 0.1  # Pattern matching typically selects ~10%
            return 1.0 / max(1, col_stats.distinct_count)
        elif operator in ('IN', 'NOT IN'):
            # Count number of values in IN clause
            values_match = re.search(r'IN\s*\((.*?)\)', condition)
            if values_match:
                num_values = len(values_match.group(1).split(','))
                return min(1.0, num_values / max(1, col_stats.distinct_count))
        return 1.0

    def estimate_index_benefit(self, table: str, index: Index, conditions: List[str]) -> float:
        """Estimate how much an index reduces the cost"""
        if table not in self.table_stats:
            return 1.0

        index_columns = set(col.name for col in index.columns)
        matching_conditions = []
        range_conditions = []
        
        for condition in conditions:
            # Extract column and operator
            col_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*([=<>!]+|LIKE|IN|NOT IN)', condition)
            if col_match:
                col_name = col_match.group(1)
                operator = col_match.group(2)
                
                if col_name in index_columns:
                    if operator == '=':
                        matching_conditions.append(condition)
                    elif operator in ['>', '<', '>=', '<=']:
                        range_conditions.append(condition)

        if not matching_conditions and not range_conditions:
            return 1.0  # No benefit if index isn't used

        # Calculate benefit based on condition types and index properties
        benefit_factor = 1.0
        
        # Equality conditions provide best benefit
        for condition in matching_conditions:
            col_name = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)', condition).group(1)
            if col_name in self.table_stats[table].column_stats:
                col_stats = self.table_stats[table].column_stats[col_name]
                selectivity = col_stats.avg_selectivity
                
                if index.is_unique and len(matching_conditions) == len(index.columns):
                    # Unique index with all columns matched provides perfect selectivity
                    benefit_factor *= 0.001  # Cost reduced to 0.1%
                else:
                    # Non-unique index or partial match
                    benefit_factor *= max(0.01, selectivity)

        # Range conditions provide moderate benefit
        for condition in range_conditions:
            col_name = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)', condition).group(1)
            if col_name in self.table_stats[table].column_stats:
                col_stats = self.table_stats[table].column_stats[col_name]
                # Range queries typically access more data
                benefit_factor *= max(0.1, col_stats.avg_selectivity * 2)

        # Consider index coverage (if all needed columns are in index)
        # TODO: Implement index coverage analysis
        
        # Apply logarithmic scaling to benefit
        # This reflects that each additional matching condition has diminishing returns
        if len(matching_conditions) + len(range_conditions) > 1:
            benefit_factor = benefit_factor * (1 + log2(len(matching_conditions) + len(range_conditions)) * 0.1)

        # Ensure reasonable bounds
        return min(0.99, max(0.001, benefit_factor))

    def estimate_join_cost(self, tables: Set[str], conditions: List[str]) -> float:
        """Estimate cost of joining tables"""
        if len(tables) <= 1:
            return 1.0

        join_cost = 0
        table_sizes = {t: self.table_stats[t].row_count for t in tables if t in self.table_stats}
        
        # Calculate cartesian product size as worst case
        total_rows = reduce(lambda x, y: x * y, table_sizes.values(), 1)
        
        # Adjust based on join conditions
        join_selectivity = 1.0
        for condition in conditions:
            if ' = ' in condition and '.' in condition:  # Likely a join condition
                join_selectivity *= 0.1  # Assume each equi-join reduces result by 90%
        
        # Final join size estimation
        estimated_result_size = max(100, total_rows * join_selectivity)
        
        # Calculate join cost considering:
        # 1. Cost of reading input tables
        # 2. Cost of processing join
        # 3. Cost of producing output
        read_cost = sum(size * 0.1 for size in table_sizes.values())  # Reading cost
        process_cost = estimated_result_size * 0.2  # Join processing cost
        output_cost = estimated_result_size * 0.1  # Result materialization cost
        
        join_cost = read_cost + process_cost + output_cost
        return max(1000.0, join_cost)  # Join operations have significant minimum overhead

    def estimate_query_cost(self, sql: str, available_indexes: Optional[List[Index]] = None) -> float:
        """Estimate the cost of executing a query with given indexes"""
        try:
            # Parse the query
            tables, conditions = self.parse_query(sql)
            
            # Start with base table access costs
            cost = self.estimate_base_cost(tables)
            
            # Add join costs if multiple tables
            if len(tables) > 1:
                cost += self.estimate_join_cost(tables, conditions)
            
            # Apply index benefits
            if available_indexes:
                index_benefit = 0
                for idx in available_indexes:
                    if idx.table in tables:
                        # Calculate potential benefit from this index
                        benefit = self.estimate_index_benefit(idx.table, idx, conditions)
                        table_cost = self.estimate_base_cost({idx.table})
                        potential_saving = table_cost * (1 - benefit)
                        index_benefit += potential_saving
                
                # Apply combined index benefits (with diminishing returns)
                if index_benefit > 0:
                    cost = max(cost * 0.1, cost - (index_benefit * 0.8))  # Preserve at least 10% of original cost

            # Add costs for additional operations
            if 'GROUP BY' in sql.upper():
                cost *= 1.5  # Group by typically increases cost by 50%
            if 'ORDER BY' in sql.upper():
                cost *= 1.3  # Order by typically increases cost by 30%
            if 'HAVING' in sql.upper():
                cost *= 1.2  # Having clause adds overhead
            
            # Apply final scaling factor based on empirical observations
            cost *= 100  # Scale up to match observed costs better
            
            return max(100.0, cost)
            
        except Exception as e:
            print(f"Error estimating query cost: {e}")
            return float('inf')

    def get_actual_cost(self, sql: str) -> float:
        """Get actual query cost from EXPLAIN output"""
        # Convert SQL to MySQL format
        mysql_sql = SQLConverter.to_mysql(sql)
        mysql_sql = SQLConverter.clean_query(mysql_sql)
        
        plan_json = self.db.get_query_plan(mysql_sql)
        if not plan_json:
            return float('inf')

        try:
            def extract_costs(node: dict) -> float:
                """Recursively extract costs from query plan"""
                if not isinstance(node, dict):
                    return 0.0

                cost = 0.0
                
                # Extract costs from current node
                cost_info = node.get('cost_info', {})
                if isinstance(cost_info, dict):
                    cost += float(cost_info.get('read_cost', 0))
                    cost += float(cost_info.get('eval_cost', 0))

                # Add row count as part of cost
                if 'rows' in node:
                    cost += float(node['rows'])

                # Process nested nodes
                for key in ['nested_loop', 'materialized', 'table', 'subqueries', 'union_result']:
                    child = node.get(key)
                    if isinstance(child, list):
                        for item in child:
                            cost += extract_costs(item)
                    elif isinstance(child, dict):
                        cost += extract_costs(child)

                return cost

            # Start cost extraction from the query block
            if isinstance(plan_json, dict) and 'query_block' in plan_json:
                return extract_costs(plan_json['query_block'])
            
            return float('inf')

        except Exception as e:
            print(f"Error extracting costs from plan: {e}")
            print(f"Plan JSON: {plan_json}")
            return float('inf')

    def validate_cost_model(self, sql: str, available_indexes: Optional[List[Index]] = None) -> Dict:
        """Compare estimated cost with actual execution plan cost"""
        try:
            estimated_cost = self.estimate_query_cost(sql, available_indexes)
            actual_cost = self.get_actual_cost(sql)

            if actual_cost == float('inf') or estimated_cost == float('inf'):
                return {'error': 'Failed to get query cost'}

            error_ratio = abs(estimated_cost - actual_cost) / actual_cost if actual_cost > 0 else float('inf')
            is_accurate = 0.5 <= estimated_cost / actual_cost <= 2.0 if actual_cost > 0 else False

            return {
                'estimated_cost': estimated_cost,
                'actual_cost': actual_cost,
                'error_ratio': error_ratio,
                'is_accurate': is_accurate
            }

        except Exception as e:
            print(f"Error validating query cost: {e}")
            return {'error': str(e)}
