import pymysql
import time
from typing import List, Tuple
from index_benefit_estimator import QueryCostEstimator, Index, Column

class ModelValidator:
    def __init__(self):
        self.estimator = QueryCostEstimator()
        self.conn = None
        
    def connect_db(self):
        """Connect to PolarDB-IMCI"""
        try:
            self.conn = pymysql.connect(
                host='172.17.0.1',
                port=22224,
                user='user1',
                password='your_password',
                database='tpch_sf1'
            )
            return True
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            return False
            
    def execute_query(self, query: str) -> Tuple[float, bool]:
        """Execute a query and measure its execution time"""
        if not self.conn:
            return -1, False
            
        cursor = self.conn.cursor()
        try:
            start_time = time.time()
            cursor.execute(query)
            cursor.fetchall()  # Ensure query completes
            end_time = time.time()
            return end_time - start_time, True
        except Exception as e:
            print(f"Query execution failed: {e}")
            return -1, False
        finally:
            cursor.close()
            
    def create_index(self, index: Index) -> bool:
        """Create an index in the database"""
        if not self.conn:
            return False
            
        cursor = self.conn.cursor()
        try:
            # Build CREATE INDEX statement
            columns = ', '.join(f"`{col.name}`" for col in index.columns)
            unique = "UNIQUE" if index.is_unique else ""
            sql = f"CREATE {unique} INDEX idx_{index.table}_{'_'.join(col.name for col in index.columns)} ON {index.table} ({columns})"
            
            cursor.execute(sql)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Failed to create index: {e}")
            return False
        finally:
            cursor.close()
            
    def drop_index(self, index: Index) -> bool:
        """Drop an index from the database"""
        if not self.conn:
            return False
            
        cursor = self.conn.cursor()
        try:
            sql = f"DROP INDEX idx_{index.table}_{'_'.join(col.name for col in index.columns)} ON {index.table}"
            cursor.execute(sql)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Failed to drop index: {e}")
            return False
        finally:
            cursor.close()
            
    def validate_index_benefit(self, query: str, index: Index) -> Tuple[float, float]:
        """Compare estimated vs actual benefit of an index"""
        # Estimate benefit using our model
        base_cost = self.estimator.estimate_query_cost(query)
        cost_with_index = self.estimator.estimate_query_cost(query, [index])
        estimated_benefit = (base_cost - cost_with_index) / base_cost if base_cost > 0 else 0
        
        # Measure actual benefit
        time_without_index, success1 = self.execute_query(query)
        if not success1:
            return 0, 0
            
        if not self.create_index(index):
            return 0, 0
            
        time_with_index, success2 = self.execute_query(query)
        self.drop_index(index)
        
        if not success2:
            return 0, 0
            
        actual_benefit = (time_without_index - time_with_index) / time_without_index if time_without_index > 0 else 0
        
        return estimated_benefit, actual_benefit

def main():
    validator = ModelValidator()
    if not validator.connect_db():
        print("Failed to connect to database")
        return
        
    # Test cases: combination of queries and indexes
    test_cases = [
        # Simple point query
        (
            'SELECT * FROM orders WHERE o_orderkey = 1',
            Index('orders', [Column('orders', 'o_orderkey')], True)
        ),
        # Range query
        (
            'SELECT * FROM lineitem WHERE l_shipdate > "1995-01-01"',
            Index('lineitem', [Column('lineitem', 'l_shipdate')])
        ),
        # Composite index test
        (
            'SELECT * FROM part WHERE p_type = "STANDARD PLATED COPPER" AND p_size = 10',
            Index('part', [Column('part', 'p_type'), Column('part', 'p_size')])
        )
    ]
    
    print("Validating index benefit estimation model...")
    print("\nResults:")
    print("=" * 80)
    print(f"{'Query':<40} {'Estimated':<12} {'Actual':<12} {'Error':<12}")
    print("-" * 80)
    
    total_error = 0
    valid_tests = 0
    
    for query, index in test_cases:
        estimated, actual = validator.validate_index_benefit(query, index)
        if estimated >= 0 and actual >= 0:
            error = abs(estimated - actual)
            total_error += error
            valid_tests += 1
            print(f"{query[:37] + '...':<40} {estimated:>10.2%} {actual:>10.2%} {error:>10.2f}")
    
    if valid_tests > 0:
        avg_error = total_error / valid_tests
        print("\nAverage estimation error:", f"{avg_error:.2f}")
    
    validator.conn.close()

if __name__ == "__main__":
    main()