import logging
import time
from selection.workload import Workload, Query
from selection.algorithms.online_bandit_algorithm import OnlineBanditAlgorithm
from selection.algorithms.extend_algorithm import ExtendAlgorithm
from selection.dbms.postgres_dbms import PostgresDatabaseConnector

# Configure logging
logging.basicConfig(level=logging.INFO)

class BenchmarkFramework:
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        self.db_connector = database_connector
        self.parameters = parameters
        self.bandit_algorithm = OnlineBanditAlgorithm(database_connector, parameters)
        self.extend_algorithm = ExtendAlgorithm(database_connector, parameters)
        self.current_workload = []
        self.periodic_interval = parameters.get("periodic_interval", 10)  # in seconds

    def simulate_workload(self, query_generator, total_time):
        """Simulate a dynamic workload for a given duration"""
        start_time = time.time()
        while time.time() - start_time < total_time:
            # Generate new query
            new_query = query_generator()
            self.current_workload.append(new_query)
            
            # Update online bandit algorithm
            self.bandit_algorithm.update_workload(new_query)
            
            # Check if it's time to run Extend algorithm
            elapsed_time = time.time() - start_time
            if elapsed_time % self.periodic_interval < 1:  # Approximate check
                workload = Workload(self.current_workload[-self.periodic_interval:])
                self.extend_algorithm._calculate_best_indexes(workload)
            
            # Sleep to simulate real-time workload
            time.sleep(1)

    def compare_algorithms(self):
        """Compare the performance of both algorithms"""
        # Calculate final costs
        bandit_indexes = list(self.bandit_algorithm.current_indexes)
        extend_indexes = self.extend_algorithm._calculate_best_indexes(Workload(self.current_workload))
        
        bandit_cost = self.bandit_algorithm.cost_evaluation.calculate_cost(
            Workload(self.current_workload), bandit_indexes
        )
        extend_cost = self.extend_algorithm.cost_evaluation.calculate_cost(
            Workload(self.current_workload), extend_indexes
        )
        
        bandit_size = sum(index.estimated_size for index in bandit_indexes)
        extend_size = sum(index.estimated_size for index in extend_indexes)
        
        logging.info("\nFinal Comparison:")
        logging.info(f"Online Bandit Algorithm:")
        logging.info(f"  Cost: {bandit_cost:.2f}")
        logging.info(f"  Storage: {b_to_mb(bandit_size):.2f}MB")
        logging.info(f"Extend Algorithm:")
        logging.info(f"  Cost: {extend_cost:.2f}")
        logging.info(f"  Storage: {b_to_mb(extend_size):.2f}MB")

def generate_random_query():
    """Generate a random query for demonstration purposes"""
    # In a real scenario, this would generate actual SQL queries
    return Query(query_id=int(time.time()), query_text="SELECT * FROM example_table WHERE example_column=?", columns=[])

if __name__ == "__main__":
    # Initialize database connector (assuming PostgreSQL configuration)
    db_connector = PostgresDatabaseConnector(
        host="172.17.0.1",
        user="wuy",
        password="wuy",
        db_name="indexselection_tpch___1"
    )
    
    # Initialize benchmark framework
    parameters = {
        "budget_MB": 500,
        "max_index_width": 3,
        "exploration_factor": 0.4,
        "window_size": 20,
        "min_utility_threshold": 0.00001,
        "periodic_interval": 10  # Run Extend every 10 seconds
    }
    benchmark = BenchmarkFramework(db_connector, parameters)
    
    # Run benchmark for 60 seconds
    benchmark.simulate_workload(generate_random_query, total_time=60)
    
    # Compare algorithms
    benchmark.compare_algorithms()