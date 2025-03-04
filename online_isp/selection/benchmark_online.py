import logging
import time
import json
import numpy as np
from typing import List, Dict

from selection.algorithms.anytime_algorithm import AnytimeAlgorithm
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from selection.algorithms.cophy_input_generation import CoPhyInputGeneration
from selection.algorithms.db2advis_algorithm import DB2AdvisAlgorithm
from selection.algorithms.dexter_algorithm import DexterAlgorithm
from selection.algorithms.drop_heuristic_algorithm import DropHeuristicAlgorithm
from selection.algorithms.extend_algorithm import ExtendAlgorithm
from selection.algorithms.relaxation_algorithm import RelaxationAlgorithm
from selection.algorithms.online_bandit_algorithm import OnlineBanditAlgorithm
from selection.benchmark import Benchmark
from selection.dbms.hana_dbms import HanaDatabaseConnector
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.query_generator import QueryGenerator
from selection.selection_algorithm import AllIndexesAlgorithm, NoIndexAlgorithm
from selection.table_generator import TableGenerator
from selection.workload import Query, Workload
from selection.workload_parser import WorkloadParser

DBMSYSTEMS = {"postgres": PostgresDatabaseConnector, "hana": HanaDatabaseConnector}


class OnlineBenchmark:
    def __init__(self, config_file: str):
        logging.getLogger().setLevel(logging.INFO)
        with open(config_file) as f:
            self.config = json.load(f)
            
        self.num_queries = self.config["num_queries"]
        # Initialize database connection
        conn_config = self.config["connection"]
        self.db_connector = PostgresDatabaseConnector(
            conn_config["dbname"],
            autocommit=True,
            host=conn_config.get("host"),
            port=conn_config.get("port"),
            user=conn_config.get("user"),
            password=conn_config.get("password", "")    
        )
        
        if WorkloadParser.is_custom_workload(self.config["benchmark_name"]):
            # use a custom workload on existing an existing database
            self.database_name = self.config["connection"]["dbname"]
            self.database_system = self.config["database_system"]
            workload_parser = WorkloadParser(
                self.database_system, self.database_name, self.config["benchmark_name"], self.num_queries
            )
            self.workload = workload_parser.execute()
            self.setup_db_connector(self.database_name, self.database_system)
            self.choose_validated_queries()
        else:
            # Generate tables if needed
            self.table_generator = TableGenerator(
                self.config["benchmark_name"],
                self.config["scale_factor"],
                self.db_connector
            )
            
            # Initialize query generator
            self.query_generator = QueryGenerator(
                self.config["benchmark_name"],
                self.config["scale_factor"],
                self.db_connector,
                self.config.get("queries"),
                self.table_generator.columns,
            )

    def setup_db_connector(self, database_name, database_system):
        if self.db_connector:
            logging.info("Create new database connector (closing old)")
            self.db_connector.close()
        self.db_connector = DBMSYSTEMS[database_system](database_name)

    def choose_validated_queries(self):
        """Filter out invalid queries from the workload"""
        validated_queries = []
        count = 0
        for query in self.workload.queries:
            try:
                self.db_connector.get_cost(query)
                query.nr = count
                validated_queries.append(query)
                count += 1
                if count == self.num_queries:
                    break
            except Exception as e:
                logging.info(f"Query {query.nr} is not valid: {e}")

        self.workload.queries = validated_queries
        logging.info(f"Validated {len(validated_queries)} queries")

    def generate_dynamic_workload(self, num_queries: int, change_frequency: int) -> List[Query]:
        """Generate a dynamic workload where query patterns change over time"""
        if hasattr(self, 'workload'):
            # Using custom workload
            logging.info(f"Using custom workload with {len(self.workload.queries)} queries")
            
            # If we have fewer queries than requested, repeat them
            if len(self.workload.queries) < num_queries:
                logging.info(f"Repeating queries to reach {num_queries} total queries")
                repeated_queries = []
                for i in range(num_queries):
                    idx = i % len(self.workload.queries)
                    query = self.workload.queries[idx]
                    # Create a copy of the query with a new ID to track it properly
                    new_query = Query(f"{query.nr}_{i//len(self.workload.queries)}", query.text)
                    new_query.columns = query.columns
                    repeated_queries.append(new_query)
                return repeated_queries
            
            # If we have more queries than needed, select a subset
            if len(self.workload.queries) >= num_queries:
                logging.info(f"Selecting {num_queries} queries from available {len(self.workload.queries)}")
                return self.workload.queries[:num_queries]
            
            return self.workload.queries
        else:
            workload = []
            current_pattern = 0
            
            for i in range(num_queries):
                if i > 0 and i % change_frequency == 0:
                    current_pattern += 1
                    
                # Modify query generation based on current pattern
                query = self.query_generator.generate_query(
                    seed=i + current_pattern * 1000  # Different seed for different patterns
                )
                workload.append(query)
                
            return workload

    def run_online_comparison(
        self,
        num_queries: int = 500,  # Reduced number of queries for faster testing
        change_frequency: int = 100,  # More frequent changes
        extend_recompute_frequency: int = 50  # More frequent recomputation
    ):
        """Compare online bandit algorithm with periodic extend algorithm and no-index baseline"""
        logging.info("Starting online index selection comparison")
        
        # Initialize metrics dictionary with recommendation times
        metrics = {
            "bandit": {
                "total_cost": 0,
                "index_changes": 0,
                "response_times": [],
                "recommendation_times": []  # Time taken to make index recommendations
            },
            "extend": {
                "total_cost": 0,
                "index_changes": 0,
                "response_times": [],
                "recommendation_times": []  # Time taken to make index recommendations
            },
            "no_index": {
                "total_cost": 0,
                "response_times": []
            }
        }
        
        # Generate dynamic workload
        workload = self.generate_dynamic_workload(num_queries, change_frequency)
        
        # Initialize algorithms
        bandit_algo = OnlineBanditAlgorithm(
            self.db_connector,
            parameters={
                "budget_MB": self.config["max_storage_mb"],
                "max_index_width": 2,
                "exploration_factor": 0.1,
                "window_size": 100,
                "min_utility_threshold": 0.01
            }
        )
        
        extend_algo = ExtendAlgorithm(
            self.db_connector,
            parameters={
                "budget_MB": self.config["max_storage_mb"],
                "max_index_width": 2
            }
        )
        
        # Initial index selection for both algorithms
        initial_workload = Workload(workload[:100])
        
        # Initial bandit selection
        start_time = time.time()
        current_bandit_indexes = set(bandit_algo._calculate_best_indexes(initial_workload))
        metrics["bandit"]["recommendation_times"].append(time.time() - start_time)
        
        # Initial extend selection
        start_time = time.time()
        current_extend_indexes = set(extend_algo._calculate_best_indexes(initial_workload))
        metrics["extend"]["recommendation_times"].append(time.time() - start_time)
        
        # Process queries
        for i, query in enumerate(workload):
            # No-index baseline
            start_time = time.time()
            no_index_cost = bandit_algo.cost_evaluation.calculate_cost(
                Workload([query]), []  # Empty list for no indexes
            )
            metrics["no_index"]["total_cost"] += no_index_cost
            metrics["no_index"]["response_times"].append(time.time() - start_time)
            
            # Online Bandit Algorithm
            start_time = time.time()
            rec_start_time = time.time()
            new_bandit_indexes = set(bandit_algo.update_workload(query))
            metrics["bandit"]["recommendation_times"].append(time.time() - rec_start_time)
            metrics["bandit"]["index_changes"] += len(new_bandit_indexes ^ current_bandit_indexes)
            current_bandit_indexes = new_bandit_indexes
            
            # Calculate cost for bandit
            bandit_cost = bandit_algo.cost_evaluation.calculate_cost(
                Workload([query]), list(current_bandit_indexes)
            )
            metrics["bandit"]["total_cost"] += bandit_cost
            metrics["bandit"]["response_times"].append(time.time() - start_time)
            
            # Extend Algorithm (periodic recomputation)
            start_time = time.time()
            if i > 0 and i % extend_recompute_frequency == 0:
                # Recompute indexes using recent queries
                recent_workload = Workload(workload[max(0, i-100):i])
                rec_start_time = time.time()
                new_extend_indexes = set(extend_algo._calculate_best_indexes(recent_workload))
                metrics["extend"]["recommendation_times"].append(time.time() - rec_start_time)
                metrics["extend"]["index_changes"] += len(new_extend_indexes ^ current_extend_indexes)
                current_extend_indexes = new_extend_indexes
            
            # Calculate cost for extend
            extend_cost = extend_algo.cost_evaluation.calculate_cost(
                Workload([query]), list(current_extend_indexes)
            )
            metrics["extend"]["total_cost"] += extend_cost
            metrics["extend"]["response_times"].append(time.time() - start_time)
            
            # Log progress
            if i > 0 and i % 100 == 0:
                logging.info(f"Processed {i} queries")
                self._log_current_metrics(metrics, i)
                
        # Final metrics
        self._log_final_metrics(metrics, num_queries)
        
        return metrics

    def _log_current_metrics(self, metrics: Dict, queries_processed: int):
        """Log current performance metrics"""
        logging.info(f"\nPerformance after {queries_processed} queries:")
        
        logging.info("No-index Baseline:")
        logging.info(f"- Total cost: {metrics['no_index']['total_cost']:.2f}")
        logging.info(f"- Avg response time: {np.mean(metrics['no_index']['response_times'])*1000:.2f}ms")
        
        logging.info("\nOnline Bandit Algorithm:")
        logging.info(f"- Total cost: {metrics['bandit']['total_cost']:.2f}")
        logging.info(f"- Index changes: {metrics['bandit']['index_changes']}")
        logging.info(f"- Avg response time: {np.mean(metrics['bandit']['response_times'])*1000:.2f}ms")
        logging.info(f"- Avg recommendation time: {np.mean(metrics['bandit']['recommendation_times'])*1000:.2f}ms")
        
        logging.info("\nPeriodic Extend Algorithm:")
        logging.info(f"- Total cost: {metrics['extend']['total_cost']:.2f}")
        logging.info(f"- Index changes: {metrics['extend']['index_changes']}")
        logging.info(f"- Avg response time: {np.mean(metrics['extend']['response_times'])*1000:.2f}ms")
        if metrics['extend']['recommendation_times']:
            logging.info(f"- Avg recommendation time: {np.mean(metrics['extend']['recommendation_times'])*1000:.2f}ms")

    def _log_final_metrics(self, metrics: Dict, num_queries: int):
        """Log final performance metrics"""
        logging.info(f"\nFinal results after {num_queries} queries:")
        
        # Calculate improvement percentages vs no-index baseline
        if metrics['no_index']['total_cost'] != 0:
            bandit_cost_improvement = (
                (metrics['no_index']['total_cost'] - metrics['bandit']['total_cost'])
                / metrics['no_index']['total_cost'] * 100
            )
            extend_cost_improvement = (
                (metrics['no_index']['total_cost'] - metrics['extend']['total_cost'])
                / metrics['no_index']['total_cost'] * 100
            )
            logging.info(f"Cost improvement vs no-index:")
            logging.info(f"  - Bandit: {bandit_cost_improvement:.2f}%")
            logging.info(f"  - Extend: {extend_cost_improvement:.2f}%")
        else:
            logging.info("Cost improvement: N/A (no baseline cost)")
            
        # Index stability comparison
        if metrics['extend']['index_changes'] != 0:
            stability_improvement = (
                (metrics['extend']['index_changes'] - metrics['bandit']['index_changes'])
                / metrics['extend']['index_changes'] * 100
            )
            logging.info(f"Index stability improvement: {stability_improvement:.2f}%")
        else:
            logging.info("Index stability improvement: N/A (no index changes)")
            
        # Response time comparison
        if all(key in metrics and metrics[key]['response_times'] for key in ['bandit', 'extend', 'no_index']):
            bandit_response = np.mean(metrics['bandit']['response_times']) * 1000
            extend_response = np.mean(metrics['extend']['response_times']) * 1000
            no_index_response = np.mean(metrics['no_index']['response_times']) * 1000
            logging.info(f"Response times (ms):")
            logging.info(f"  - No-index: {no_index_response:.2f}")
            logging.info(f"  - Bandit: {bandit_response:.2f}")
            logging.info(f"  - Extend: {extend_response:.2f}")
            
        # Recommendation time comparison
        if all(key in metrics and metrics[key].get('recommendation_times') for key in ['bandit', 'extend']):
            bandit_rec_time = np.mean(metrics['bandit']['recommendation_times']) * 1000
            extend_rec_time = np.mean(metrics['extend']['recommendation_times']) * 1000
            logging.info(f"\nRecommendation times (ms):")
            logging.info(f"  - Bandit: {bandit_rec_time:.2f}")
            logging.info(f"  - Extend: {extend_rec_time:.2f}")
            
        # Log absolute metrics
        logging.info("\nAbsolute metrics:")
        logging.info(f"No-index total cost: {metrics['no_index']['total_cost']:.2f}")
        logging.info(f"Bandit total cost: {metrics['bandit']['total_cost']:.2f}")
        logging.info(f"Extend total cost: {metrics['extend']['total_cost']:.2f}")
        logging.info(f"Bandit index changes: {metrics['bandit']['index_changes']}")
        logging.info(f"Extend index changes: {metrics['extend']['index_changes']}")

if __name__ == "__main__":
    # Example configuration
    config = {
        "benchmark_name": "tpch",
        "scale_factor": 1,
        "max_storage_mb": 500,
        "queries": None  # Use default query templates
    }
    
    # Save config to file
    with open("online_benchmark_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run benchmark
    benchmark = OnlineBenchmark("online_benchmark_config.json")
    metrics = benchmark.run_online_comparison(
        num_queries=1000,
        change_frequency=200,  # Change query pattern every 200 queries
        extend_recompute_frequency=100  # Recompute extend algorithm every 100 queries
    )