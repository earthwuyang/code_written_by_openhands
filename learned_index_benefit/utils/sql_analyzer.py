import re
from typing import List, Dict, Set, Tuple
from models.data_models import Column

class SQLAnalyzer:
    @staticmethod
    def parse_sql(sql: str) -> Dict:
        """Parse SQL query and extract key components"""
        sql = sql.strip()
        result = {
            'tables': set(),
            'columns': set(),
            'conditions': [],
            'joins': [],
            'group_by': [],
            'order_by': [],
            'aggregates': [],
            'has_subquery': False
        }

        # Extract tables and their aliases
        from_clause = SQLAnalyzer._extract_from_clause(sql)
        if from_clause:
            tables_and_aliases = SQLAnalyzer._parse_table_references(from_clause)
            result['tables'] = set(table for table, _ in tables_and_aliases)
            result['table_aliases'] = dict(tables_and_aliases)

        # Extract WHERE conditions
        where_clause = SQLAnalyzer._extract_where_clause(sql)
        if where_clause:
            result['conditions'] = SQLAnalyzer._parse_conditions(where_clause)

        # Extract JOIN conditions
        result['joins'] = SQLAnalyzer._extract_joins(sql)

        # Extract GROUP BY and ORDER BY
        result['group_by'] = SQLAnalyzer._extract_group_by(sql)
        result['order_by'] = SQLAnalyzer._extract_order_by(sql)

        # Check for subqueries
        result['has_subquery'] = SQLAnalyzer._has_subquery(sql)

        return result

    @staticmethod
    def _extract_from_clause(sql: str) -> str:
        """Extract the FROM clause from SQL query"""
        match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _parse_table_references(from_clause: str) -> List[Tuple[str, str]]:
        """Parse table references and their aliases from FROM clause"""
        tables_and_aliases = []
        # Split on comma for multiple tables
        parts = from_clause.split(',')
        for part in parts:
            # Handle JOIN syntax
            join_parts = re.split(r'\s+(?:JOIN|INNER JOIN|LEFT JOIN|RIGHT JOIN|OUTER JOIN)\s+', part)
            for table_ref in join_parts:
                table_ref = table_ref.strip()
                if table_ref:
                    # Extract table name and optional alias
                    match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)?', table_ref)
                    if match:
                        table_name = match.group(1).lower()
                        alias = match.group(2).lower() if match.group(2) else table_name
                        tables_and_aliases.append((table_name, alias))
        return tables_and_aliases

    @staticmethod
    def _extract_where_clause(sql: str) -> str:
        """Extract the WHERE clause from SQL query"""
        match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _parse_conditions(where_clause: str) -> List[Dict]:
        """Parse conditions from WHERE clause"""
        conditions = []
        # Split on AND, handling parentheses
        parts = SQLAnalyzer._split_on_and(where_clause)
        for part in parts:
            part = part.strip()
            if part:
                # Parse each condition
                match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)\s*([=<>!]+|LIKE|IN|NOT IN)\s*(.*)', part)
                if match:
                    conditions.append({
                        'column': match.group(1),
                        'operator': match.group(2),
                        'value': match.group(3).strip("'\"")
                    })
        return conditions

    @staticmethod
    def _split_on_and(where_clause: str) -> List[str]:
        """Split WHERE clause on AND, respecting parentheses"""
        parts = []
        current_part = []
        paren_count = 0
        
        tokens = where_clause.split()
        for token in tokens:
            paren_count += token.count('(') - token.count(')')
            if token.upper() == 'AND' and paren_count == 0:
                if current_part:
                    parts.append(' '.join(current_part))
                    current_part = []
            else:
                current_part.append(token)
        
        if current_part:
            parts.append(' '.join(current_part))
        
        return parts

    @staticmethod
    def _extract_joins(sql: str) -> List[Dict]:
        """Extract JOIN conditions from SQL query"""
        joins = []
        join_pattern = r'((?:INNER|LEFT|RIGHT|OUTER)?\s*JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)?\s+ON\s+(.*?)(?:(?:INNER|LEFT|RIGHT|OUTER)?\s*JOIN|WHERE|GROUP BY|ORDER BY|LIMIT|$)'
        
        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            joins.append({
                'type': match.group(1).strip().upper(),
                'table': match.group(2).lower(),
                'alias': (match.group(3) or match.group(2)).lower(),
                'condition': match.group(4).strip()
            })
        return joins

    @staticmethod
    def _extract_group_by(sql: str) -> List[str]:
        """Extract GROUP BY columns"""
        match = re.search(r'GROUP BY\s+(.*?)(?:HAVING|ORDER BY|LIMIT|$)', sql, re.IGNORECASE)
        if match:
            return [col.strip() for col in match.group(1).split(',')]
        return []

    @staticmethod
    def _extract_order_by(sql: str) -> List[Dict]:
        """Extract ORDER BY columns and directions"""
        match = re.search(r'ORDER BY\s+(.*?)(?:LIMIT|$)', sql, re.IGNORECASE)
        if not match:
            return []

        order_by = []
        for item in match.group(1).split(','):
            item = item.strip()
            if item:
                parts = item.split()
                order_by.append({
                    'column': parts[0],
                    'direction': parts[1].upper() if len(parts) > 1 else 'ASC'
                })
        return order_by

    @staticmethod
    def _has_subquery(sql: str) -> bool:
        """Check if SQL contains subqueries"""
        # Look for SELECT within parentheses
        return bool(re.search(r'\(\s*SELECT\s', sql, re.IGNORECASE))

    @staticmethod
    def analyze_query_complexity(sql: str) -> Dict:
        """Analyze query complexity based on various factors"""
        analysis = SQLAnalyzer.parse_sql(sql)
        
        complexity = {
            'score': 0,  # Base complexity score
            'factors': []
        }

        # Number of tables (joins)
        num_tables = len(analysis['tables'])
        if num_tables > 1:
            complexity['score'] += (num_tables - 1) * 2
            complexity['factors'].append(f"Joins: {num_tables - 1}")

        # Number of conditions
        num_conditions = len(analysis['conditions'])
        if num_conditions > 0:
            complexity['score'] += num_conditions
            complexity['factors'].append(f"Conditions: {num_conditions}")

        # GROUP BY
        if analysis['group_by']:
            complexity['score'] += 2
            complexity['factors'].append("Has GROUP BY")

        # ORDER BY
        if analysis['order_by']:
            complexity['score'] += 1
            complexity['factors'].append("Has ORDER BY")

        # Subqueries
        if analysis['has_subquery']:
            complexity['score'] += 5
            complexity['factors'].append("Has subqueries")

        # Categorize complexity
        if complexity['score'] <= 3:
            complexity['level'] = 'Simple'
        elif complexity['score'] <= 7:
            complexity['level'] = 'Moderate'
        elif complexity['score'] <= 12:
            complexity['level'] = 'Complex'
        else:
            complexity['level'] = 'Very Complex'

        return complexity