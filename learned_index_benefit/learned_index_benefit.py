import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json
import re
import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass
import pymysql
from index_benefit_estimator import Index, Column

class LearnedIndexBenefitEstimator:
    def __init__(self, debug=False):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.conn = None
        self.debug = debug
        self._debug_count = 0
        self._table_sizes = {}  # Cache for table sizes
        self.connect_db()
        if self.debug:
            print("Initialized LearnedIndexBenefitEstimator in debug mode")

    def connect_db(self):
        """Connect to PolarDB-IMCI database"""
        try:
            self.conn = pymysql.connect(
                host='172.17.0.1',
                port=22224,
                user='user1',
                password='your_password',
                database='tpch_sf1',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise

    def _extract_query_features(self, query: str, indexes: List[Index] = None) -> Dict:
        """Extract features from a SQL query"""
        # Initialize with default values
        features = {
            'num_tables': 0, 'num_joins': 0, 'num_conditions': 0,
            'has_group_by': 0, 'has_order_by': 0, 'has_aggregation': 0,
            'num_indexes': len(indexes) if indexes else 0,
            'num_unique_indexes': len([idx for idx in (indexes or []) if idx.is_unique]),
            'avg_index_columns': 0, 'max_index_columns': 0,
            'index_coverage_score': 0,
            'max_table_size': 0, 'total_table_size': 0, 'avg_table_size': 0,
            'num_eq_conditions': 0, 'num_range_conditions': 0,
            'num_like_conditions': 0, 'num_in_conditions': 0,
            'indexed_conditions': 0,
            'num_pk_fk_joins': 0, 'num_many_many_joins': 0,
            'indexed_joins': 0, 'max_join_depth': 0,
            'is_point_query': 0, 'is_range_query': 0,
            'is_join_query': 0, 'is_aggregation_query': 0
        }
        
        try:
            # Extract tables and their aliases
            tables_matches = re.findall(r'(?i)FROM\s+`?(\w+)`?(?:\s+(?:AS\s+)?(\w+))?|JOIN\s+`?(\w+)`?(?:\s+(?:AS\s+)?(\w+))?', query)
            tables = {}  # Map of alias -> table name
            for match in tables_matches:
                if match[0]:  # FROM clause
                    tables[match[1] if match[1] else match[0]] = match[0]
                elif match[2]:  # JOIN clause
                    tables[match[3] if match[3] else match[2]] = match[2]
            
            features['num_tables'] = len(tables)
            
            # Get table sizes (using class-level cache)
            cursor = self.conn.cursor()
            total_size = 0
            max_size = 0
            for table in tables.values():
                if table not in self._table_sizes:
                    try:
                        cursor.execute(f"SELECT COUNT(*) as cnt FROM `{table}`")
                        self._table_sizes[table] = cursor.fetchone()['cnt']
                        if self.debug and self._debug_count < 3:
                            print(f"Caching size for table {table}: {self._table_sizes[table]} rows")
                    except Exception as e:
                        if self.debug:
                            print(f"Error getting size for table {table}: {e}")
                        self._table_sizes[table] = 0
                size = self._table_sizes[table]
                total_size += size
                max_size = max(max_size, size)
            
            features['max_table_size'] = max_size
            features['total_table_size'] = total_size
            features['avg_table_size'] = total_size / max(1, len(tables))
            
            # Extract conditions
            conditions = re.findall(r'WHERE(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
            if conditions:
                cond_text = conditions[0]
                features['num_eq_conditions'] = len(re.findall(r'=(?!=)', cond_text))
                features['num_range_conditions'] = len(re.findall(r'[<>]|<=|>=', cond_text))
                features['num_like_conditions'] = len(re.findall(r'LIKE', cond_text, re.IGNORECASE))
                features['num_in_conditions'] = len(re.findall(r'IN\s*\(', cond_text, re.IGNORECASE))
                features['num_conditions'] = (features['num_eq_conditions'] + features['num_range_conditions'] +
                                           features['num_like_conditions'] + features['num_in_conditions'])
                
                # Check if conditions use indexed columns
                if indexes:
                    indexed_columns = {f"{idx.table}.{col.name}" for idx in indexes for col in idx.columns}
                    features['indexed_conditions'] = sum(1 for col in indexed_columns 
                                                      if re.search(rf'\b{col}\b\s*[=<>]', cond_text, re.IGNORECASE))
            
            # Extract joins
            joins = re.findall(r'JOIN.*?ON\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
            features['num_joins'] = len(joins)
            features['is_join_query'] = int(features['num_joins'] > 0)
            
            if joins and indexes:
                # Count joins on indexed columns
                indexed_columns = {f"{idx.table}.{col.name}" for idx in indexes for col in idx.columns}
                features['indexed_joins'] = sum(1 for join_cond in joins 
                                             for col in indexed_columns 
                                             if re.search(rf'\b{col}\b', join_cond, re.IGNORECASE))
            
            # Query type features
            features['is_point_query'] = int(features['num_eq_conditions'] > 0 and features['num_range_conditions'] == 0)
            features['is_range_query'] = int(features['num_range_conditions'] > 0)
            features['has_group_by'] = int(bool(re.search(r'GROUP BY', query, re.IGNORECASE)))
            features['has_order_by'] = int(bool(re.search(r'ORDER BY', query, re.IGNORECASE)))
            features['has_aggregation'] = int(bool(re.search(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(', query, re.IGNORECASE)))
            features['is_aggregation_query'] = features['has_aggregation']
            
            # Calculate index features
            if indexes:
                col_counts = [len(idx.columns) for idx in indexes]
                features['avg_index_columns'] = sum(col_counts) / len(indexes)
                features['max_index_columns'] = max(col_counts)
                
                # Calculate index coverage score
                coverage_score = sum(
                    (len(idx.columns) * 0.5) *  # Base weight for multi-column indexes
                    (1.5 if idx.is_unique else 1.0) *  # Weight for unique indexes
                    (2.0 if any(re.search(rf'\b{idx.table}\.{col.name}\b', query, re.IGNORECASE) 
                              for col in idx.columns) else 1.0)  # Weight for used indexes
                    for idx in indexes
                )
                features['index_coverage_score'] = coverage_score
            
            # Add debug information only for the first few queries
            if self.debug:
                if self._debug_count < 3:
                    print("\nExtracted features:")
                    for k, v in sorted(features.items()):
                        if v != 0:  # Only show non-zero features
                            print(f"  {k}: {v}")
                    print(f"Query: {query[:100]}...")
                elif self._debug_count == 3:
                    print("\n(Debug output suppressed for remaining queries)")
                self._debug_count += 1
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            print(f"Query: {query[:200]}...")
            return features

    def _get_actual_cost(self, query: str) -> float:
        """Get actual query cost from EXPLAIN output"""
        try:
            # Clean and normalize the query
            query = query.strip()
            if not query:
                return float('inf')
            
            # Convert double quotes to backticks and normalize whitespace
            query = query.replace('"', '`')
            query = re.sub(r'\s+', ' ', query)
            
            cursor = self.conn.cursor()
            explain_query = f"EXPLAIN FORMAT=JSON {query}"
            
            try:
                cursor.execute(explain_query)
                plan = cursor.fetchone()
            except Exception as e:
                print(f"Error executing EXPLAIN: {e}")
                print(f"Query: {query[:200]}...")
                return float('inf')
                
            if not plan or not any(key.lower() == 'explain' for key in plan.keys()):
                print(f"Invalid EXPLAIN output format")
                return float('inf')
            
            # Find the EXPLAIN key (case-insensitive)
            explain_key = next(key for key in plan.keys() if key.lower() == 'explain')
            plan_json = plan[explain_key]
            
            try:
                plan_data = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
            except json.JSONDecodeError as e:
                print(f"Error parsing EXPLAIN JSON: {e}")
                return float('inf')
            
            def extract_cost(node):
                """Recursively extract cost from plan node"""
                if not node:
                    return 0
                    
                cost = 0
                if isinstance(node, dict):
                    # Extract cost_info
                    cost_info = node.get('cost_info', {})
                    if isinstance(cost_info, dict):
                        cost += float(cost_info.get('read_cost', 0))
                        cost += float(cost_info.get('eval_cost', 0))
                    
                    # Extract rows_examined_per_scan
                    if 'rows_examined_per_scan' in node:
                        try:
                            cost += float(node['rows_examined_per_scan'])
                        except (ValueError, TypeError):
                            pass
                    
                    # Extract rows
                    if 'rows' in node:
                        try:
                            cost += float(node.get('rows', 0))
                        except (ValueError, TypeError):
                            pass
                    
                    # Process nested nodes
                    for key in ['nested_loop', 'table', 'materialized', 'subqueries']:
                        value = node.get(key)
                        if isinstance(value, list):
                            for child in value:
                                cost += extract_cost(child)
                        elif isinstance(value, dict):
                            cost += extract_cost(value)
                    
                    # Process attached_subqueries
                    for subquery in node.get('attached_subqueries', []):
                        cost += extract_cost(subquery)
                            
                elif isinstance(node, list):
                    for child in node:
                        cost += extract_cost(child)
                        
                return cost
            
            # Extract cost from query block
            if isinstance(plan_data, dict) and 'query_block' in plan_data:
                cost = extract_cost(plan_data['query_block'])
            else:
                cost = extract_cost(plan_data)
                
            return max(1.0, cost)
            
        except Exception as e:
            print(f"Unexpected error in _get_actual_cost: {e}")
            print(f"Query: {query[:200]}...")
            return float('inf')

    def generate_training_data(self, query_files: List[str], num_samples: int = 1000) -> pd.DataFrame:
        """Generate training data from query files"""
        data = []
        
        # Read and clean queries
        def clean_query(query):
            if not query:
                return ""
                
            # Remove comments
            query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
            query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
            
            # Clean whitespace
            query = re.sub(r'\s+', ' ', query)
            query = query.strip()
            
            # Add semicolon if missing
            if not query.strip().endswith(';'):
                query = query.strip() + ';'
                
            # Basic validation
            if query.count('(') != query.count(')'):
                if self.debug:
                    print("Query has unmatched parentheses, skipping...")
                return ""
                
            if query.count("'") % 2 != 0:
                if self.debug:
                    print("Query has unmatched quotes, skipping...")
                return ""
            
            # Convert schema.column format to proper quoting
            def quote_identifier(match):
                schema = match.group(1)
                column = match.group(2)
                return f"`{schema}`.`{column}`"
            
            # First handle schema.column format with double quotes
            query = re.sub(r'"(\w+)"."(\w+)"', quote_identifier, query)
            
            # Then handle any remaining double-quoted identifiers
            query = re.sub(r'"(\w+)"', r'`\1`', query)
            
            # Handle string literals
            def handle_string(match):
                s = match.group(1)
                s = s.replace("'", "\\'")  # Escape single quotes
                return f"'{s}'"
            query = re.sub(r"'([^']*)'", handle_string, query)
            
            # Handle numeric literals (round to 6 decimal places)
            def handle_number(match):
                try:
                    num = float(match.group(0))
                    if num.is_integer():
                        return str(int(num))
                    return f"{num:.6f}"
                except ValueError:
                    return match.group(0)
            query = re.sub(r'-?\d+\.\d+', handle_number, query)
            
            # Remove any duplicate backticks
            query = re.sub(r'`+', '`', query)
            query = re.sub(r'`([^`]+)`', r'`\1`', query)
            
            if self.debug:
                print("\nCleaned query:")
                print(f"Original: {query[:200]}")
                print(f"Cleaned:  {query[:200]}")
            
            return query

        def validate_query(query):
            """Validate if a query is executable"""
            # Initialize debug counter if needed
            if self.debug and not hasattr(self, '_validate_count'):
                self._validate_count = 0
            
            # Show validation details only for first few queries
            show_debug = self.debug and (not hasattr(self, '_validate_count') or self._validate_count < 3)
            if self.debug:
                self._validate_count += 1
                if show_debug:
                    print("\nValidating query:")
                    print(query[:100], "..." if len(query) > 100 else "")
            
            # Check query length
            if len(query) > 2000:  # Skip very long queries
                if show_debug:
                    print(f"Query too long ({len(query)} chars), skipping...")
                return False
                
            # Check if query is complete
            if query.count('(') != query.count(')'):
                if show_debug:
                    print("Query has unmatched parentheses, skipping...")
                return False
                
            # Basic query validation
            if not query.strip().upper().startswith('SELECT'):
                if show_debug:
                    print("Query must start with SELECT")
                return False
                
            if not query.strip().endswith(';'):
                if show_debug:
                    print("Query must end with semicolon")
                return False
                
            try:
                cursor = self.conn.cursor()
                
                # Try EXPLAIN first
                explain_query = f"EXPLAIN FORMAT=JSON {query}"
                if show_debug:
                    print("\nExecuting EXPLAIN:")
                    print(explain_query[:100], "..." if len(explain_query) > 100 else "")
                    
                try:
                    cursor.execute(explain_query)
                    plan = cursor.fetchone()
                    if not plan:
                        if show_debug:
                            print("EXPLAIN returned no results")
                        return False
                        
                    explain_key = next((k for k in plan.keys() if k.lower() == 'explain'), None)
                    if not explain_key:
                        if show_debug:
                            print("No EXPLAIN key found in result")
                            print("Available keys:", list(plan.keys()))
                        return False
                        
                    if show_debug:
                        print("EXPLAIN successful")
                        
                except Exception as e:
                    if show_debug:
                        print(f"EXPLAIN failed: {str(e)[:100]}...")
                    return False
                
                # Try executing with LIMIT 1
                test_query = f"{query.rstrip(';')} LIMIT 1;"
                if show_debug:
                    print("\nTesting execution:")
                    print(test_query[:100], "..." if len(test_query) > 100 else "")
                    
                try:
                    cursor.execute(test_query)
                    cursor.fetchall()  # Consume results
                    if show_debug:
                        print("Execution successful")
                    return True
                except Exception as e:
                    if show_debug:
                        print(f"Execution failed: {str(e)[:100]}...")
                    return False
                    
            except Exception as e:
                if show_debug:
                    print(f"Validation failed: {str(e)[:100]}...")
                return False

        queries = []
        total_queries = 0
        valid_queries = 0
        
        for file in query_files:
            try:
                print(f"\nProcessing file: {file}")
                with open(file, 'r') as f:
                    content = f.read()
                    # Split by semicolon and process each query
                    raw_queries = [q.strip() for q in content.split(';') if q.strip()]
                    total_queries += len(raw_queries)
                    
                    for query in raw_queries:
                        if not query.upper().startswith(('SELECT', 'UPDATE', 'DELETE', 'INSERT')):
                            if self.debug:
                                print(f"Skipping non-DML query: {query[:100]}...")
                            continue
                            
                        try:
                            cleaned_query = clean_query(query)
                            if not cleaned_query:
                                continue
                                
                            if validate_query(cleaned_query):
                                queries.append(cleaned_query)
                                valid_queries += 1
                                if self.debug and valid_queries % 10 == 0:
                                    print(f"Processed {valid_queries} valid queries...")
                            
                        except Exception as e:
                            if self.debug:
                                print(f"Error processing query: {str(e)[:100]}...")
                            continue
                            
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
                
        print(f"\nQuery processing summary:")
        print(f"Total queries found: {total_queries}")
        print(f"Valid queries: {valid_queries}")
        print(f"Invalid/skipped queries: {total_queries - valid_queries}")

        print(f"\nSuccessfully loaded {len(queries)} valid queries")
        
        # Generate training data with different index combinations
        for query in queries:
            try:
                # Base case - no indexes
                features = self._extract_query_features(query)
                cost = self._get_actual_cost(query)
                if cost != float('inf'):
                    features['actual_cost'] = cost
                    data.append(features.copy())
                
                # Extract tables involved in the query
                tables_matches = re.findall(r'(?i)FROM\s+`?(\w+)`?|JOIN\s+`?(\w+)`?', query)
                tables = set()
                for match in tables_matches:
                    tables.update(t for t in match if t)
                
                # Generate single-column index combinations
                for table in tables:
                    cursor = self.conn.cursor()
                    try:
                        # Get columns for this table
                        cursor.execute(f"SHOW COLUMNS FROM `{table}`")
                        columns = [row['Field'] for row in cursor.fetchall()]
                        
                        # Try each column as an index
                        for col in columns:
                            idx = Index(table, [Column(table, col)])
                            features = self._extract_query_features(query, [idx])
                            cost = self._get_actual_cost(query)
                            if cost != float('inf'):
                                features['actual_cost'] = cost
                                data.append(features.copy())
                        
                        # Try some multi-column indexes (limit to 2 columns for now)
                        for i in range(len(columns)):
                            for j in range(i + 1, len(columns)):
                                idx = Index(table, [
                                    Column(table, columns[i]),
                                    Column(table, columns[j])
                                ])
                                features = self._extract_query_features(query, [idx])
                                cost = self._get_actual_cost(query)
                                if cost != float('inf'):
                                    features['actual_cost'] = cost
                                    data.append(features.copy())
                    except Exception as e:
                        print(f"Error processing table {table}: {str(e)[:100]}...")
                        continue
                    
            except Exception as e:
                print(f"Error processing query: {str(e)[:100]}...")
                continue
        
        # Sample queries
        if len(queries) > num_samples:
            queries = np.random.choice(queries, num_samples, replace=False)
        
        # Generate different index combinations
        tables = ['customer', 'orders', 'lineitem', 'part', 'partsupp', 'supplier', 'nation', 'region']
        columns = {
            'customer': ['c_custkey', 'c_nationkey'],
            'orders': ['o_orderkey', 'o_custkey'],
            'lineitem': ['l_orderkey', 'l_partkey', 'l_suppkey'],
            'part': ['p_partkey'],
            'partsupp': ['ps_partkey', 'ps_suppkey'],
            'supplier': ['s_suppkey', 's_nationkey'],
            'nation': ['n_nationkey', 'n_regionkey'],
            'region': ['r_regionkey']
        }
        
        # Generate index combinations
        for query in queries:
            # Base case - no indexes
            features = self._extract_query_features(query)
            cost = self._get_actual_cost(query)
            if cost != float('inf'):
                features['actual_cost'] = cost
                data.append(features.copy())
            
            # Single column indexes
            for table in tables:
                for col in columns[table]:
                    idx = Index(table, [Column(table, col)])
                    features = self._extract_query_features(query, [idx])
                    cost = self._get_actual_cost(query)
                    if cost != float('inf'):
                        features['actual_cost'] = cost
                        data.append(features.copy())
            
            # Multi-column indexes (up to 2 columns)
            for table in tables:
                if len(columns[table]) >= 2:
                    for i in range(len(columns[table])):
                        for j in range(i+1, len(columns[table])):
                            idx = Index(table, [
                                Column(table, columns[table][i]),
                                Column(table, columns[table][j])
                            ])
                            features = self._extract_query_features(query, [idx])
                            cost = self._get_actual_cost(query)
                            if cost != float('inf'):
                                features['actual_cost'] = cost
                                data.append(features.copy())
        
        return pd.DataFrame(data)

    def train_model(self, data: pd.DataFrame):
        """Train the model on generated data"""
        # Separate features and target
        X = data.drop('actual_cost', axis=1)
        y = data['actual_cost']
        self.feature_names = X.columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"Train R² score: {train_score:.4f}")
        print(f"Test R² score: {test_score:.4f}")
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 most important features:")
        print(importances.head(10))

    def predict_cost(self, query: str, indexes: List[Index] = None) -> float:
        """Predict query cost using the trained model"""
        if not self.model:
            raise Exception("Model not trained yet")
            
        # Extract features
        features = self._extract_query_features(query, indexes)
        features_df = pd.DataFrame([features])
        
        # Scale features
        features_scaled = self.scaler.transform(features_df[self.feature_names])
        
        # Predict
        predicted_cost = self.model.predict(features_scaled)[0]
        return max(1.0, predicted_cost)

def main():
    try:
        # Initialize estimator with debug mode
        print("Initializing index benefit estimator...")
        estimator = LearnedIndexBenefitEstimator(debug=True)  # Enable debug output
        
        # Generate training data
        print("\nGenerating training data...")
        query_files = [
            'data/workloads/tpch_sf1/TP_queries.sql'  # Start with just TP queries for testing
        ]
        
        # Verify files exist
        for file in query_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Query file not found: {file}")
        
        print(f"Reading queries from {len(query_files)} files...")
        # Use a smaller sample size for initial testing
        num_samples = 10  # Reduce sample size for testing
        print(f"Using {num_samples} samples for testing...")
        
        try:
            # Test database connection first
            cursor = estimator.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            print("Database connection successful")
            
            # Generate training data
            print("\nGenerating training data...")
            data = estimator.generate_training_data(query_files, num_samples=num_samples)
            print(f"\nGenerated {len(data)} training samples")
            
            if len(data) == 0:
                print("\nError: No valid training data generated")
                print("Possible issues:")
                print("1. Query parsing/validation failures")
                print("2. Database connection issues")
                print("3. Invalid SQL syntax in workload files")
                raise ValueError("No valid training data generated")
            elif len(data) < num_samples:
                print(f"\nWarning: Only generated {len(data)} samples out of {num_samples} requested")
                print("Continuing with reduced dataset...")
            
            # Print sample of the training data
            print("\nSample of training features:")
            sample_df = pd.DataFrame(data).head()
            print(sample_df[['num_tables', 'num_joins', 'num_conditions', 'index_coverage_score', 'actual_cost']].to_string())
            
        except Exception as e:
            print(f"\nError generating training data: {str(e)}")
            raise
            
        # Train model
        print("\nTraining model...")
        estimator.train_model(data)
        
        # Test with simple queries first
        print("\nTesting model predictions...")
        
        # Test Case 1: Simple COUNT query
        print("\n1. Simple COUNT query:")
        query1 = "SELECT COUNT(*) FROM orders;"
        cost1 = estimator.predict_cost(query1)
        print(f"Query: {query1}")
        print(f"Predicted cost (no indexes): {cost1:.2f}")
        
        # Test Case 2: Simple query with WHERE clause
        print("\n2. Simple query with WHERE clause:")
        query2 = "SELECT o_orderkey, o_orderdate FROM orders WHERE o_orderdate = '1993-07-01';"
        
        # Without index
        cost2_no_idx = estimator.predict_cost(query2)
        print(f"Query: {query2}")
        print(f"Predicted cost (no indexes): {cost2_no_idx:.2f}")
        
        # With index on o_orderdate
        idx1 = Index('orders', [Column('orders', 'o_orderdate')])
        cost2_with_idx = estimator.predict_cost(query2, [idx1])
        print(f"Predicted cost (with o_orderdate index): {cost2_with_idx:.2f}")
        print(f"Index benefit: {cost2_no_idx - cost2_with_idx:.2f}")
        
        # Test Case 3: Simple JOIN query
        print("\n3. Simple JOIN query:")
        query3 = """
        SELECT o.o_orderkey, l.l_linenumber 
        FROM orders o 
        JOIN lineitem l ON o.o_orderkey = l.l_orderkey 
        WHERE o.o_orderdate = '1993-07-01';
        """
        
        # Without indexes
        cost3_no_idx = estimator.predict_cost(query3)
        print(f"Query: {query3}")
        print(f"Predicted cost (no indexes): {cost3_no_idx:.2f}")
        
        # With indexes
        idx2 = Index('lineitem', [Column('lineitem', 'l_orderkey')])
        cost3_with_idx = estimator.predict_cost(query3, [idx1, idx2])
        print(f"Predicted cost (with indexes): {cost3_with_idx:.2f}")
        print(f"Index benefit: {cost3_no_idx - cost3_with_idx:.2f}")
        
        # Print feature importance
        print("\nFeature Importance Summary:")
        importances = pd.DataFrame({
            'feature': estimator.feature_names,
            'importance': estimator.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importances.head(10).to_string())
        
        print("\nModel evaluation completed successfully!")
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()