import pymysql

def test_connection():
    try:
        print("Attempting to connect to database...")
        conn = pymysql.connect(
            host='172.17.0.1',
            port=22224,
            user='user1',
            password='your_password',
            database='tpch_sf1'
        )
        print("Successfully connected to database!")
        
        cursor = conn.cursor()
        
        # Test basic query
        print("\nTesting basic query...")
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE()")
        count = cursor.fetchone()[0]
        print(f"Number of tables in database: {count}")
        
        # Test EXPLAIN
        print("\nTesting EXPLAIN capability...")
        cursor.execute("EXPLAIN FORMAT=JSON SELECT * FROM region LIMIT 1")
        explain_result = cursor.fetchone()
        print("EXPLAIN result:", explain_result)
        
        cursor.close()
        conn.close()
        print("\nDatabase connection test completed successfully!")
        
    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    test_connection()