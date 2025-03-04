import logging
import os
import platform
import re
import subprocess

from selection.workload import Query


class QueryGenerator:
    def __init__(self, benchmark_name, scale_factor, db_connector, query_ids, columns):
        self.scale_factor = scale_factor
        self.benchmark_name = benchmark_name
        self.db_connector = db_connector
        self.queries = []
        self.query_ids = query_ids
        # All columns in current database/schema
        self.columns = columns

        self.generate()

    def filter_queries(self, query_ids):
        self.queries = [query for query in self.queries if query.nr in query_ids]

    def add_new_query(self, query_id, query_text):
        if not self.db_connector:
            logging.info("{}:".format(self))
            logging.error("No database connector to validate queries")
            raise Exception("database connector missing")
        query_text = self.db_connector.update_query_text(query_text)
        query = Query(query_id, query_text)
        self._validate_query(query)
        self._store_indexable_columns(query)
        self.queries.append(query)

    def _validate_query(self, query):
        try:
            self.db_connector.get_plan(query)
        except Exception as e:
            self.db_connector.rollback()
            logging.error("{}: {}".format(self, e))

    def _store_indexable_columns(self, query):
        # Look for column references in the query
        for column in self.columns:
            # Check for both unqualified and qualified column names
            column_patterns = [
                r'\b' + re.escape(column.name) + r'\b',  # Unqualified
                r'\b' + re.escape(f"{column.table.name}.{column.name}") + r'\b'  # Qualified
            ]
            if any(re.search(pattern, query.text.lower()) for pattern in column_patterns):
                query.columns.append(column)

    def _generate_tpch(self):
        logging.info("Generating TPC-H Queries")
        self._run_make()
        # Using default parameters (`-d`)
        queries_string = self._run_command(
            ["./qgen", "-c", "-d", "-s", str(self.scale_factor)], return_output=True
        )
        for query in queries_string.split("Query (Q"):
            query_id_and_text = query.split(")\n", 1)
            if len(query_id_and_text) == 2:
                query_id, text = query_id_and_text
                query_id = int(query_id)
                if self.query_ids and query_id not in self.query_ids:
                    continue
                text = text.replace("\t", "")
                self.add_new_query(query_id, text)
        logging.info("Queries generated")

    def _generate_tpcds(self):
        logging.info("Generating TPC-DS Queries")
        self._run_make()
        # dialects: ansi, db2, netezza, oracle, sqlserver
        command = [
            "./dsqgen",
            "-DIRECTORY",
            "../query_templates",
            "-INPUT",
            "../query_templates/templates.lst",
            "-DIALECT",
            "netezza",
            "-QUALIFY",
            "Y",
            "-OUTPUT_DIR",
            "../..",
        ]
        self._run_command(command)
        file_path = os.path.dirname(os.path.abspath(__file__))
        with open(file_path + "/../query_0.sql", "r") as file:
            queries_string = file.read()
        for query_string in queries_string.split("-- start query"):
            id_and_text = query_string.split(".tpl\n", 1)
            if len(id_and_text) != 2:
                continue
            query_id = int(id_and_text[0].split("using template query")[-1])
            if self.query_ids and query_id not in self.query_ids:
                continue
            query_text = id_and_text[1]
            query_text = self._update_tpcds_query_text(query_text)
            self.add_new_query(query_id, query_text)

    # This manipulates TPC-DS specific queries to work in more DBMSs
    def _update_tpcds_query_text(self, query_text):
        query_text = query_text.replace(") returns", ") as returns")
        replaced_string = "case when lochierarchy = 0"
        if replaced_string in query_text:
            new_string = re.search(
                r"grouping\(.*\)\+" r"grouping\(.*\) " r"as lochierarchy", query_text
            ).group(0)
            new_string = new_string.replace(" as lochierarchy", "")
            new_string = "case when " + new_string + " = 0"
            query_text = query_text.replace(replaced_string, new_string)
        return query_text

    def _run_make(self):
        if "qgen" not in self._files() and "dsqgen" not in self._files():
            logging.info(f"Running {self.make_command} in {self.directory}")
            self._run_command(self.make_command)
        else:
            logging.debug("No need to run make")

    def _run_command(self, command, return_output=False, shell=False):
        env = os.environ.copy()
        env["DSS_QUERY"] = "queries"
        p = subprocess.Popen(
            command,
            cwd=self.directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=shell,
            env=env,
        )
        with p.stdout:
            output_string = p.stdout.read().decode("utf-8")
        p.wait()
        if return_output:
            return output_string
        else:
            logging.debug("[SUBPROCESS OUTPUT] " + output_string)

    def _files(self):
        return os.listdir(self.directory)

    def generate_query(self, seed=None):
        """Generate a new query with optional seed for reproducibility"""
        import random
        if seed is not None:
            random.seed(seed)
            
        # If we have existing queries, modify one randomly
        if self.queries:
            base_query = random.choice(self.queries)
            modified_query = self._modify_query(base_query)
            return modified_query
        else:
            # Generate initial queries if none exist
            self.generate()
            return random.choice(self.queries)
            
    def _modify_query(self, base_query):
        """Modify a query to create a variant"""
        import random
        
        # Create a copy of the query
        query_text = base_query.text
        
        # Possible modifications:
        # 1. Change constants in WHERE clauses
        # 2. Change aggregation functions
        # 3. Change ORDER BY direction
        
        # 1. Change constants, but preserve date format
        # Find date literals first
        date_pattern = r"date '\d{4}-\d{2}-\d{2}'"
        dates = re.findall(date_pattern, query_text)
        date_positions = [(m.start(), m.end()) for m in re.finditer(date_pattern, query_text)]
        
        # Find numbers that are not part of dates
        number_pattern = r'\d+(?:\.\d+)?'
        for match in re.finditer(number_pattern, query_text):
            start, end = match.span()
            
            # Skip if this number is part of a date
            is_in_date = False
            for date_start, date_end in date_positions:
                if date_start <= start <= date_end:
                    is_in_date = True
                    break
                    
            # Only modify numbers that are part of WHERE clauses or LIMIT
            if (not is_in_date and 
                random.random() < 0.2 and  # Reduce modification probability
                ('WHERE' in query_text or 'LIMIT' in query_text)):
                number = match.group()
                try:
                    new_value = float(number)
                    # Keep the modification reasonable
                    new_value = new_value * random.uniform(0.8, 1.2)
                    
                    # Format the number appropriately
                    if '.' in number:
                        # For decimal numbers, keep similar precision
                        original_decimals = len(number.split('.')[1])
                        new_value = round(new_value, original_decimals)
                        formatted_value = f"{new_value:.{original_decimals}f}"
                        # Remove trailing zeros while keeping at least one decimal
                        if '.' in formatted_value:
                            formatted_value = formatted_value.rstrip('0').rstrip('.')
                            if formatted_value == '':
                                formatted_value = '0'
                    else:
                        # For integers, round to nearest integer
                        formatted_value = str(int(round(new_value)))
                        
                    query_text = (
                        query_text[:start] +
                        formatted_value +
                        query_text[end:]
                    )
                except ValueError:
                    continue
                
        # 2. Change aggregations
        agg_functions = ['SUM', 'AVG', 'MAX', 'MIN']
        for func in agg_functions:
            if func in query_text and random.random() < 0.2:
                new_func = random.choice([f for f in agg_functions if f != func])
                query_text = query_text.replace(func, new_func)
                
        # 3. Change ORDER BY
        if 'ORDER BY' in query_text:
            if random.random() < 0.3:
                if 'DESC' in query_text:
                    query_text = query_text.replace('DESC', 'ASC')
                else:
                    query_text = query_text.replace('ASC', 'DESC')
                    
        # Create new query object
        new_query = Query(base_query.nr, query_text)
        new_query.columns = base_query.columns.copy()
        
        return new_query

    def generate(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        if self.benchmark_name == "tpch":
            self.directory = file_path + "/../tpch-kit/dbgen"
            # DBMS in tpch-kit dbgen Makefile:
            # INFORMIX, DB2, TDAT (Teradata),
            # SQLSERVER, SYBASE, ORACLE, VECTORWISE, POSTGRESQL
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("MACHINE=MACOS")

            self._generate_tpch()
        elif self.benchmark_name == "tpcds":
            self.directory = file_path + "/../tpcds-kit/tools"
            self.make_command = ["make"]
            if platform.system() == "Darwin":
                self.make_command.append("OS=MACOS")

            self._generate_tpcds()
        else:
            raise NotImplementedError("only tpch/tpcds implemented.")
