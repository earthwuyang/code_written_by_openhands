import psycopg2

# Connect to the database
conn = psycopg2.connect(
    dbname="tpch_sf1",
    user="wuy",
    password="wuy",
    host="172.17.0.1",
    port=5432
)

cur = conn.cursor()

# Get all tables
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
""")
tables = cur.fetchall()

print("Tables in database:")
for table in tables:
    table_name = table[0]
    print(f"\nTable: {table_name}")
    
    # Get columns for each table
    cur.execute(f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = '{table_name}'
    """)
    columns = cur.fetchall()
    
    print("Columns:")
    for col in columns:
        print(f"  {col[0]}: {col[1]}")

cur.close()
conn.close()