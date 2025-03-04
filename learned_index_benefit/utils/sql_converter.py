class SQLConverter:
    @staticmethod
    def to_mysql(sql: str) -> str:
        """Convert SQL with double quotes to MySQL format with backticks"""
        # Remove any existing backticks first
        sql = sql.replace('`', '')
        
        # Handle table.column patterns
        parts = []
        current = 0
        in_quotes = False
        quote_start = -1
        
        for i, char in enumerate(sql):
            if char == '"':
                if not in_quotes:
                    in_quotes = True
                    quote_start = i
                else:
                    in_quotes = False
                    quoted_part = sql[quote_start+1:i]
                    # Check if it's a table.column pattern
                    if '.' in quoted_part:
                        table, column = quoted_part.split('.')
                        parts.append(sql[current:quote_start] + f'`{table}`.`{column}`')
                    else:
                        parts.append(sql[current:quote_start] + f'`{quoted_part}`')
                    current = i + 1

        if current < len(sql):
            parts.append(sql[current:])

        converted_sql = ''.join(parts)
        
        # Clean up any remaining quotes and normalize whitespace
        converted_sql = ' '.join(converted_sql.split())
        
        return converted_sql

    @staticmethod
    def clean_query(sql: str) -> str:
        """Clean and normalize SQL query"""
        # Remove comments
        sql = ' '.join(line for line in sql.split('\n') 
                      if not line.strip().startswith('--'))
        
        # Normalize whitespace
        sql = ' '.join(sql.split())
        
        # Ensure query ends with semicolon
        if not sql.strip().endswith(';'):
            sql = sql + ';'
            
        return sql