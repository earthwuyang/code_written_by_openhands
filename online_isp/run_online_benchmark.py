import logging
import random
from selection.benchmark_online import OnlineBenchmark

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s'
    )
    
    # Update database name in config
    import json
    with open("online_benchmark_config_new.json", "r") as f:
        config = json.load(f)
    
    seed = config["seed"]
    random.seed(seed)

    # Initialize benchmark with updated config
    benchmark = OnlineBenchmark("online_benchmark_config_new.json")
    
    # Run comparison with different scenarios:
    # 1. Gradual workload change
    logging.info("\nScenario 1: Gradual workload change")
    metrics_gradual = benchmark.run_online_comparison(
        num_queries=config["num_queries"],
        change_frequency=50,  # Change pattern every 50 queries
        extend_recompute_frequency=20  # Recompute extend every 20 queries
    )
    
    # # 2. Rapid workload change
    # logging.info("\nScenario 2: Rapid workload change")
    # metrics_rapid = benchmark.run_online_comparison(
    #     num_queries=200,
    #     change_frequency=20,  # Change pattern every 20 queries
    #     extend_recompute_frequency=10   # Recompute extend every 10 queries
    # )
    
    # # 3. Long-term stability test
    # logging.info("\nScenario 3: Long-term stability test")
    # metrics_stable = benchmark.run_online_comparison(
    #     num_queries=400,
    #     change_frequency=100,  # Change pattern every 100 queries
    #     extend_recompute_frequency=40  # Recompute extend every 40 queries
    # )