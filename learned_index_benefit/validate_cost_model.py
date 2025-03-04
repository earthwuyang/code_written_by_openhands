from models.data_models import Index, Column
from index_benefit_estimator import QueryCostEstimator
from database.db_connector import DatabaseConnector
from utils.sql_analyzer import SQLAnalyzer
from typing import List, Dict
import json
from datetime import datetime
import statistics

class CostModelValidator:
    def __init__(self):
        self.db = DatabaseConnector(
            host='172.17.0.1',
            port=22224,
            user='user1',
            password='your_password',
            database='tpch_sf1'
        )
        self.estimator = QueryCostEstimator()
        self.sql_analyzer = SQLAnalyzer()

    def validate_queries(self, queries: List[str], indexes: List[Index] = None) -> Dict:
        """Validate cost model accuracy across multiple queries"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(queries),
            'successful_validations': 0,
            'accuracy_stats': {
                'error_ratios': [],
                'accurate_estimates': 0
            },
            'query_details': [],
            'skipped_queries': 0  # Track queries skipped due to zero cost
        }

        for i, query in enumerate(queries):
            # try:
            validation = self.estimator.validate_cost_model(query, indexes)

            # Skip queries with zero actual cost
            if validation['actual_cost'] == 0:
                results['skipped_queries'] += 1
                continue

            query_detail = {
                'query_id': i + 1,
                'query': query[:100] + '...' if len(query) > 100 else query,
                'estimated_cost': validation['estimated_cost'],
                'actual_cost': validation['actual_cost'],
                'error_ratio': validation['error_ratio'],
                'is_accurate': validation['is_accurate']
            }

            # Add more detailed analysis
            if validation['estimated_cost'] > validation['actual_cost']:
                query_detail['issue'] = 'overestimation'
                query_detail['factor'] = validation['estimated_cost'] / validation['actual_cost']
            else:
                query_detail['issue'] = 'underestimation'
                query_detail['factor'] = validation['actual_cost'] / validation['estimated_cost']

            results['query_details'].append(query_detail)
            results['successful_validations'] += 1
            results['accuracy_stats']['error_ratios'].append(validation['error_ratio'])
            if validation['is_accurate']:
                results['accuracy_stats']['accurate_estimates'] += 1

            # except Exception as e:
            #     print(f"Error validating query {i+1}: {e}")

        # Calculate aggregate statistics
        if results['accuracy_stats']['error_ratios']:
            results['accuracy_stats']['avg_error'] = statistics.mean(
                results['accuracy_stats']['error_ratios']
            )
            results['accuracy_stats']['median_error'] = statistics.median(
                results['accuracy_stats']['error_ratios']
            )
            results['accuracy_stats']['accuracy_rate'] = (
                results['accuracy_stats']['accurate_estimates'] /
                results['successful_validations']
            ) if results['successful_validations'] > 0 else 0

        return results

    def generate_report(self, results: Dict):
        """Generate a detailed validation report"""
        print("\nCost Model Validation Report")
        print("=" * 80)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Total Queries: {results['total_queries']}")
        print(f"Queries with Valid Costs: {results['successful_validations']}")
        print(f"Queries Skipped (Zero Cost): {results.get('skipped_queries', 0)}")

        if results['successful_validations'] > 0:
            print("\nAccuracy Statistics:")
            print("-" * 40)
            
            # Count estimation issues
            overestimations = sum(1 for d in results['query_details'] if d.get('issue') == 'overestimation')
            underestimations = sum(1 for d in results['query_details'] if d.get('issue') == 'underestimation')
            
            print(f"Overestimations: {overestimations} ({overestimations/len(results['query_details'])*100:.1f}%)")
            print(f"Underestimations: {underestimations} ({underestimations/len(results['query_details'])*100:.1f}%)")
            print(f"Average Error Ratio: {results['accuracy_stats'].get('avg_error', float('inf')):.2f}")
            print(f"Median Error Ratio: {results['accuracy_stats'].get('median_error', float('inf')):.2f}")
            print(f"Accuracy Rate: {results['accuracy_stats'].get('accuracy_rate', 0):.1%}")

            print("\nDetailed Query Analysis:")
            print("-" * 100)
            print(f"{'Query ID':<8} {'Est. Cost':<12} {'Act. Cost':<12} {'Error':<10} {'Issue':<15} {'Factor':<10}")
            print("-" * 100)

            for detail in results['query_details']:
                print(f"{detail['query_id']:<8} {detail['estimated_cost']:>10.0f} "
                      f"{detail['actual_cost']:>10.0f} {detail['error_ratio']:>8.2f} "
                      f"{detail.get('issue', 'unknown'):<15} {detail.get('factor', 0):>8.1f}x")

            # Analyze problematic queries
            print("\nProblematic Queries (Error Ratio > 2.0):")
            print("-" * 80)
            problematic = [d for d in results['query_details'] if d['error_ratio'] > 2.0]
            if problematic:
                # Sort by error ratio
                problematic.sort(key=lambda x: x['error_ratio'], reverse=True)
                for detail in problematic[:3]:  # Show top 3 worst cases
                    print(f"\nQuery {detail['query_id']} ({detail.get('issue', 'unknown')}, "
                          f"Factor: {detail.get('factor', 0):.1f}x):")
                    print(f"Query: {detail['query']}")
                    print(f"Estimated Cost: {detail['estimated_cost']:.0f}")
                    print(f"Actual Cost: {detail['actual_cost']:.0f}")
                    print(f"Error Ratio: {detail['error_ratio']:.2f}")
            else:
                print("No significantly inaccurate estimates found.")

            # Enhanced recommendations based on error patterns
            print("\nRecommendations:")
            print("-" * 80)
            accuracy_rate = results['accuracy_stats'].get('accuracy_rate', 0)
            
            if overestimations > underestimations * 2:
                print("- Cost model tends to overestimate significantly")
                print("- Consider reducing cost factors in estimation")
                print("- Review selectivity estimation for complex predicates")
            elif underestimations > overestimations * 2:
                print("- Cost model tends to underestimate significantly")
                print("- Consider increasing cost factors in estimation")
                print("- Review join cost estimation")
                
            if accuracy_rate < 0.5:
                print("- Cost model needs significant improvement")
                print("- Consider recalibrating selectivity estimates")
                print("- Review statistics collection process")
            elif accuracy_rate < 0.8:
                print("- Cost model shows moderate accuracy")
                print("- Focus on improving estimates for complex queries")
                print("- Consider adding more detailed statistics")
            else:
                print("- Cost model shows good accuracy")
                print("- Monitor for specific query patterns that show higher error rates")
                print("- Consider fine-tuning for edge cases")

def main():
    validator = CostModelValidator()

    # Read test queries
    print("Reading test queries...")
    def read_queries(file_path):
        queries = []
        current_query = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('--'):
                    if current_query:
                        query = ' '.join(current_query)
                        # Only add if it's a valid query
                        if 'SELECT' in query.upper():
                            queries.append(query)
                        current_query = []
                    continue
                # Handle query termination
                if line.endswith(';'):
                    current_query.append(line[:-1])  # Remove semicolon
                    query = ' '.join(current_query)
                    if 'SELECT' in query.upper():
                        queries.append(query)
                    current_query = []
                else:
                    current_query.append(line)
        # Handle last query if any
        if current_query:
            query = ' '.join(current_query)
            if 'SELECT' in query.upper():
                queries.append(query)
        print(f"Read {len(queries)} queries from {file_path}")
        return queries

    tp_queries = read_queries('data/workloads/tpch_sf1/TP_queries.sql')
    ap_queries = read_queries('data/workloads/tpch_sf1/workload_100k_s1_group_order_by_more_complex.sql')

    # TODO:generate more diversified combination of indexes instead of only current manually specifeid indexes, use permutations of columns to generate indexes
    # Define test indexes
    test_indexes = [
        Index('orders', [Column('orders', 'o_orderkey')], True),
        Index('customer', [Column('customer', 'c_custkey')], True),
        Index('lineitem', [Column('lineitem', 'l_orderkey')]),
        Index('part', [Column('part', 'p_partkey')], True),
        Index('supplier', [Column('supplier', 's_suppkey')], True)
    ]

    # Use a reasonable number of queries for validation
    max_queries = 30  # Validate with 100 queries from each type

    # Validate TP queries
    print(f"\nValidating TP queries (first {max_queries})...")
    tp_results = validator.validate_queries(tp_queries[:max_queries], test_indexes)
    validator.generate_report(tp_results)

    # Validate AP queries
    print(f"\nValidating AP queries (first {max_queries})...")
    ap_results = validator.validate_queries(ap_queries[:max_queries], test_indexes)
    validator.generate_report(ap_results)

    # Save results
    with open('cost_model_validation.json', 'w') as f:
        json.dump({
            'tp_results': tp_results,
            'ap_results': ap_results
        }, f, indent=2)

if __name__ == "__main__":
    main()