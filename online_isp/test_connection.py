import psycopg2

# PostgreSQL connection details
DB_HOST = "172.17.0.1"  # Use host IP if this doesn't work
DB_PORT = "5432"
DB_NAME = "tpch_sf1"  # Replace with your DB name
DB_USER = "wuy"      # Replace with your username
DB_PASSWORD = "wuy"  # Replace with your password

try:
    # Attempt connection
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    # Create a cursor
    cur = conn.cursor()
    cur.execute("SELECT version();")
    db_version = cur.fetchone()
    
    print("✅ Connected to PostgreSQL successfully!")
    print(f"Database version: {db_version[0]}")
    
    # Close the connection
    cur.close()
    conn.close()
    
except Exception as e:
    print("❌ Failed to connect to PostgreSQL")
    print(f"Error: {e}")
