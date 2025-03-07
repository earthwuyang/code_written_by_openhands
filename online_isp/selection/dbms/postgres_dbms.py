import logging
import re

import psycopg2

from selection.database_connector import DatabaseConnector


class PostgresDatabaseConnector(DatabaseConnector):
    def __init__(self, db_name, autocommit=False, host=None, port=None, user=None, password=None):
        DatabaseConnector.__init__(self, db_name, autocommit=autocommit)
        self.db_system = "postgres"
        self._connection = None
        self.host = host
        self.port = port
        self.user = user
        self.password = password

        if not self.db_name:
            self.db_name = "postgres"
        self.create_connection()

        self.set_random_seed()

        logging.debug("Postgres connector created: {}".format(db_name))

    def create_connection(self):
        if self._connection:
            self.close()
            
        conn_params = {
            "dbname": self.db_name
        }
        if self.host:
            conn_params["host"] = self.host
        if self.port:
            conn_params["port"] = self.port
        if self.user:
            conn_params["user"] = self.user
        if self.password:
            conn_params["password"] = self.password
            
        self._connection = psycopg2.connect(**conn_params)
        self._connection.autocommit = self.autocommit
        self._cursor = self._connection.cursor()
        
        # Set search_path to public schema
        self.exec_only("SET search_path TO public")
        
        # Analyze tables to update statistics
        self.exec_only("ANALYZE")
        
        self.commit()

    def enable_simulation(self):
        try:
            # Try to create extension
            self.exec_only("create extension if not exists hypopg")
            self.commit()
        except Exception as e:
            logging.error(f"Error enabling hypopg: {e}")
            # Try to drop and recreate if there was an error
            self.exec_only("drop extension if exists hypopg")
            self.commit()
            self.exec_only("create extension hypopg")
            self.commit()

    def database_names(self):
        result = self.exec_fetch("select datname from pg_database", False)
        return [x[0] for x in result]

    # Updates query syntax to work in PostgreSQL
    def update_query_text(self, text):
        text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
        text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)
        text = self._add_alias_subquery(text)
        return text

    # PostgreSQL requires an alias for subqueries
    def _add_alias_subquery(self, query_text):
        text = query_text.lower()
        positions = []
        for match in re.finditer(r"((from)|,)[  \n]*\(", text):
            counter = 1
            pos = match.span()[1]
            while counter > 0:
                char = text[pos]
                if char == "(":
                    counter += 1
                elif char == ")":
                    counter -= 1
                pos += 1
            next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
            if next_word[0] in [")", ","] or next_word in [
                "limit",
                "group",
                "order",
                "where",
            ]:
                positions.append(pos)
        for pos in sorted(positions, reverse=True):
            query_text = query_text[:pos] + " as alias123 " + query_text[pos:]
        return query_text

    def create_database(self, database_name):
        self.exec_only("create database {}".format(database_name))
        logging.info("Database {} created".format(database_name))

    def import_data(self, table, path, delimiter="|"):
        with open(path, "r") as file:
            self._cursor.copy_from(file, table, sep=delimiter, null="")

    def indexes_size(self):
        # Returns size in bytes
        statement = (
            "select sum(pg_indexes_size(table_name::text)) from "
            "(select table_name from information_schema.tables "
            "where table_schema='public') as all_tables"
        )
        result = self.exec_fetch(statement)
        return result[0]

    def drop_database(self, database_name):
        statement = f"DROP DATABASE {database_name};"
        self.exec_only(statement)

        logging.info(f"Database {database_name} dropped")

    def create_statistics(self):
        logging.info("Postgres: Run `analyze`")
        self.commit()
        self._connection.autocommit = True
        self.exec_only("analyze")
        self._connection.autocommit = self.autocommit

    def set_random_seed(self, value=0.17):
        logging.info(f"Postgres: Set random seed `SELECT setseed({value})`")
        self.exec_only(f"SELECT setseed({value})")

    def supports_index_simulation(self):
        if self.db_system == "postgres":
            return True
        return False

    def _simulate_index(self, index):
        table_name = index.table()
        # Use column names without table qualification for CREATE INDEX
        column_names = [col.name for col in index.columns]
        joined_columns = ", ".join(column_names)
        statement = (
            "select * from hypopg_create_index( "
            f"'create index on {table_name} "
            f"({joined_columns})')"
        )
        logging.debug(f"Simulating index with statement: {statement}")
        try:
            result = self.exec_fetch(statement)
            logging.debug(f"Index simulation result: {result}")
            return result
        except Exception as e:
            logging.error(f"Error simulating index: {e}")
            logging.error(f"Table: {table_name}, Columns: {column_names}")
            # Check if table and columns exist
            self.exec_only(f"SELECT {joined_columns} FROM {table_name} LIMIT 0")
            raise

    def _drop_simulated_index(self, oid):
        statement = f"select * from hypopg_drop_index({oid})"
        result = self.exec_fetch(statement)

        assert result[0] is True, f"Could not drop simulated index with oid = {oid}."

    def create_index(self, index):
        table_name = index.table()
        # Use column names without table qualification for CREATE INDEX
        column_names = [col.name for col in index.columns]
        joined_columns = ", ".join(column_names)
        statement = (
            f"create index {index.index_idx()} "
            f"on {table_name} ({joined_columns})"
        )
        self.exec_only(statement)
        size = self.exec_fetch(
            f"select relpages from pg_class c " f"where c.relname = '{index.index_idx()}'"
        )
        size = size[0]
        index.estimated_size = size * 8 * 1024

    def drop_indexes(self):
        logging.info("Dropping indexes")
        stmt = "select indexname from pg_indexes where schemaname='public'"
        indexes = self.exec_fetch(stmt, one=False)
        for index in indexes:
            index_name = index[0]
            drop_stmt = "drop index {}".format(index_name)
            logging.debug("Dropping index {}".format(index_name))
            self.exec_only(drop_stmt)

    # PostgreSQL expects the timeout in milliseconds
    def exec_query(self, query, timeout=None, cost_evaluation=False):
        # Committing to not lose indexes after timeout
        if not cost_evaluation:
            self._connection.commit()
        query_text = self._prepare_query(query)
        if timeout:
            set_timeout = f"set statement_timeout={timeout}"
            self.exec_only(set_timeout)
        statement = f"explain (analyze, buffers, format json) {query_text}"
        try:
            plan = self.exec_fetch(statement, one=True)[0][0]["Plan"]
            result = plan["Actual Total Time"], plan
        except Exception as e:
            logging.error(f"{query.nr}, {e}")
            self._connection.rollback()
            result = None, self._get_plan(query)
        # Disable timeout
        self._cursor.execute("set statement_timeout = 0")
        self._cleanup_query(query)
        return result

    def exec_fetchall(self, query):
        self._cursor.execute(query)
        return self._cursor.fetchall()

    def _cleanup_query(self, query):
        for query_statement in query.text.split(";"):
            if "drop view" in query_statement:
                self.exec_only(query_statement)
                self.commit()

    def _get_cost(self, query):
        query_plan = self._get_plan(query)
        total_cost = query_plan["Total Cost"]
        return total_cost

    def _get_plan(self, query):
        query_text = self._prepare_query(query)
        
        # Drop any existing revenue views first
        try:
            self.exec_only("DROP VIEW IF EXISTS revenue0 CASCADE;")
            self.exec_only("DROP VIEW IF EXISTS revenue1 CASCADE;")
            self.commit()
        except:
            self.rollback()

        # Replace revenue0/revenue1 subqueries directly in the query
        revenue_subquery = """
            SELECT l_suppkey as supplier_no,
            SUM(l_extendedprice * (1 - l_discount)) as total_revenue
            FROM lineitem
            WHERE l_shipdate >= date '1996-01-01'
            AND l_shipdate < date '1996-01-01' + interval '3 month'
            GROUP BY l_suppkey
        """
        
        # Replace revenue0 and revenue1 references with the subquery
        query_text = query_text.replace("revenue0", f"({revenue_subquery}) as revenue0")
        query_text = query_text.replace("revenue1", f"({revenue_subquery}) as revenue1")
        
        # Get the query plan
        statement = f"explain (format json) {query_text}"
        query_plan = self.exec_fetch(statement)[0][0]["Plan"]
        
        return query_plan

    def number_of_indexes(self):
        statement = """select count(*) from pg_indexes
                       where schemaname = 'public'"""
        result = self.exec_fetch(statement)
        return result[0]

    def table_exists(self, table_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_tables
            WHERE tablename = '{table_name}');"""
        result = self.exec_fetch(statement)
        return result[0]

    def database_exists(self, database_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_database
            WHERE datname = '{database_name}');"""
        result = self.exec_fetch(statement)
        return result[0]
