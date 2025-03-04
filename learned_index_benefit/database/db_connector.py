import pymysql
from typing import Dict, Optional
from models.data_models import TableStatistics, ColumnStatistics

class DatabaseConnector:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.conn = None
        self.connect()

    def connect(self):
        """Connect to PolarDB-IMCI database with retries"""
        max_retries = 3
        retry_delay = 1
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting database connection (attempt {attempt + 1}/{max_retries})...")
                self.conn = pymysql.connect(**self.config)
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

    def collect_statistics(self) -> Dict[str, TableStatistics]:
        """Collect table and column statistics from the database"""
        if not self.conn:
            raise Exception("No database connection available")
            
        cursor = self.conn.cursor()
        table_stats = {}
        
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
                table_name = row['TABLE_NAME'].lower()
                rows = row['TABLE_ROWS'] or 0
                avg_row_len = row['AVG_ROW_LENGTH'] or 0
                data_len = row['DATA_LENGTH'] or 0
                idx_len = row['INDEX_LENGTH'] or 0
                
                # Convert lengths to MB for display
                data_mb = data_len / (1024 * 1024) if data_len else 0
                idx_mb = idx_len / (1024 * 1024) if idx_len else 0
                
                print(f"{table_name:<20} {rows:<12} {avg_row_len:<12} "
                      f"{data_mb:>9.2f} MB {idx_mb:>9.2f} MB")
                
                table_stats[table_name] = TableStatistics(
                    name=table_name,
                    row_count=rows,
                    avg_row_length=avg_row_len,
                    data_length=data_len,
                    index_length=idx_len,
                    column_stats={}
                )
                
                # Analyze table to update statistics
                print(f"\nAnalyzing table {table_name}...")
                cursor.execute(f"ANALYZE TABLE {table_name}")
                
                # Get column information
                cursor.execute(f"SHOW COLUMNS FROM {table_name}")
                columns = cursor.fetchall()
                
                print(f"\nColumn statistics for {table_name}:")
                print(f"{'Column':<20} {'Type':<15} {'Distinct':<10} {'Selectivity':<12}")
                print("-" * 60)
                
                for col in columns:
                    col_name = col['Field'].lower()
                    col_type = col['Type'].lower()
                    
                    try:
                        # Get basic column statistics
                        cursor.execute(f"""
                            SELECT 
                                COUNT(DISTINCT `{col_name}`) as distinct_count,
                                COUNT(*) as total_count
                            FROM {table_name}
                        """)
                        stats = cursor.fetchone()
                        
                        if stats:
                            distinct_count = int(stats['distinct_count']) if stats['distinct_count'] is not None else 0
                            total_count = int(stats['total_count']) if stats['total_count'] is not None else 0
                            selectivity = distinct_count / total_count if total_count > 0 else 1
                            
                            col_stats = ColumnStatistics(
                                name=col_name,
                                data_type=col_type,
                                distinct_count=distinct_count,
                                total_count=total_count,
                                avg_selectivity=selectivity
                            )
                            
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
                                    FROM {table_name}
                                """)
                                result = cursor.fetchone()
                                
                                if result['min_val'] is not None:
                                    col_stats.min_value = float(result['min_val'])
                                    col_stats.max_value = float(result['max_val'])
                                    col_stats.avg_value = float(result['avg_val'])
                                    print(f"  → Range: {col_stats.min_value} to {col_stats.max_value}, "
                                          f"Avg: {col_stats.avg_value:.2f}")
                            
                            table_stats[table_name].column_stats[col_name] = col_stats
                            
                    except Exception as e:
                        print(f"Error collecting statistics for {table_name}.{col_name}: {e}")
            
            print("\nStatistics collection completed.")
            return table_stats
            
        except Exception as e:
            print(f"Error loading statistics: {e}")
            raise
        finally:
            cursor.close()

    def get_query_plan(self, sql: str) -> Optional[dict]:
        """Get query execution plan in JSON format"""
        if not self.conn:
            return None
            
        cursor = self.conn.cursor()
        try:
            explain_sql = f"EXPLAIN FORMAT=JSON {sql}"
            # print(f"Executing: {explain_sql}")
            cursor.execute(explain_sql)
            plan = cursor.fetchone()
            
            if not plan or 'EXPLAIN' not in plan:
                print("No EXPLAIN plan returned")
                return None
                
            try:
                import json
                return json.loads(plan['EXPLAIN'])
            except json.JSONDecodeError as e:
                print(f"Failed to parse EXPLAIN plan: {e}")
                print(f"Raw plan: {plan['EXPLAIN']}")
                return None
                
        except Exception as e:
            print(f"Error getting query plan: {e}")
            print(f"SQL: {sql}")
            return None
        finally:
            cursor.close()