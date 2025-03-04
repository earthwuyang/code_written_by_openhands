from index_benefit_estimator import QueryCostEstimator, Index, Column
from typing import List, Dict, Tuple
import json
from datetime import datetime

class ImplementationValidator:
    def __init__(self):
        self.estimator = QueryCostEstimator()
        
        # Define implementation phases
        self.phases = {
            1: {
                'name': 'High-Impact, Low-Cost Indexes',
                'indexes': [
                    Index('part', [Column('part', 'p_container')], False),
                    Index('customer', [Column('customer', 'c_custkey')], True)
                ]
            },
            2: {
                'name': 'Medium-Impact Indexes',
                'indexes': [
                    Index('orders', [Column('orders', 'o_totalprice')], False),
                    Index('part', [Column('part', 'p_size')], False)
                ]
            },
            3: {
                'name': 'High-Cost Index',
                'indexes': [
                    Index('lineitem', [Column('lineitem', 'l_suppkey')], False)
                ]
            }
        }
        
    def estimate_storage_impact(self, indexes: List[Index]) -> Dict[str, float]:
        """Estimate storage impact of indexes in MB"""
        impact = {}
        for idx in indexes:
            table_size = self.estimator.table_sizes.get(idx.table, 0)
            storage_mb = (table_size * 20 * len(idx.columns)) / (1024 * 1024)
            impact[str(idx)] = storage_mb
        return impact
        
    def analyze_phase(self, phase: int, queries: List[str]) -> Dict:
        """Analyze impact of implementing a specific phase"""
        # Get indexes for current and previous phases
        current_indexes = []
        for p in range(1, phase + 1):
            current_indexes.extend(self.phases[p]['indexes'])
            
        # Analyze performance
        base_cost = sum(self.estimator.estimate_query_cost(q) for q in queries)
        cost_with_indexes = sum(self.estimator.estimate_query_cost(q, current_indexes) for q in queries)
        
        # Calculate improvements
        total_improvement = ((base_cost - cost_with_indexes) / base_cost * 100) if base_cost > 0 else 0
        
        # Analyze individual queries
        query_impacts = []
        for i, query in enumerate(queries):
            cost_before = self.estimator.estimate_query_cost(query)
            cost_after = self.estimator.estimate_query_cost(query, current_indexes)
            improvement = ((cost_before - cost_after) / cost_before * 100) if cost_before > 0 else 0
            if improvement > 0:
                query_impacts.append({
                    'query_id': i + 1,
                    'improvement': improvement
                })
        
        # Calculate storage impact
        storage_impact = self.estimate_storage_impact(self.phases[phase]['indexes'])
        total_storage = sum(storage_impact.values())
        
        return {
            'phase': phase,
            'phase_name': self.phases[phase]['name'],
            'total_improvement': total_improvement,
            'queries_improved': len(query_impacts),
            'storage_impact_mb': total_storage,
            'query_impacts': query_impacts,
            'index_storage': storage_impact
        }
        
    def validate_implementation(self, queries: List[str]):
        """Validate the complete implementation plan"""
        print("Index Implementation Validation Report")
        print("=" * 80)
        print(f"Total Queries: {len(queries)}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("\nPhase Analysis:")
        print("-" * 80)
        
        cumulative_improvement = 0
        cumulative_storage = 0
        
        for phase in range(1, 4):
            results = self.analyze_phase(phase, queries)
            cumulative_storage += results['storage_impact_mb']
            
            print(f"\nPhase {phase}: {results['phase_name']}")
            print(f"{'='*40}")
            print(f"Improvement: {results['total_improvement']:.1f}%")
            print(f"Queries Improved: {results['queries_improved']}")
            print(f"Storage Impact: {results['storage_impact_mb']:.1f} MB")
            print("\nIndexes in this phase:")
            for idx, storage in results['index_storage'].items():
                print(f"- {idx}: {storage:.1f} MB")
            
            if results['query_impacts']:
                print("\nTop query improvements:")
                for impact in sorted(results['query_impacts'], 
                                   key=lambda x: x['improvement'], 
                                   reverse=True)[:3]:
                    print(f"Query {impact['query_id']}: {impact['improvement']:.1f}%")
            
            print(f"\nCumulative storage: {cumulative_storage:.1f} MB")
            print(f"Risk assessment: {'Low' if cumulative_storage < 100 else 'Medium' if cumulative_storage < 200 else 'High'}")
        
        print("\nFinal Assessment:")
        print("-" * 80)
        final_results = self.analyze_phase(3, queries)
        print(f"Total improvement after all phases: {final_results['total_improvement']:.1f}%")
        print(f"Total storage impact: {cumulative_storage:.1f} MB")
        print(f"Total queries improved: {final_results['queries_improved']}")
        
        # Provide recommendations
        print("\nRecommendations:")
        if final_results['total_improvement'] < 5:
            print("- Consider revising index selection - improvements below target")
        if cumulative_storage > 200:
            print("- Consider phasing implementation more gradually due to high storage impact")
        if final_results['queries_improved'] < len(queries) * 0.1:
            print("- Review query patterns - low percentage of queries improved")

def main():
    validator = ImplementationValidator()
    
    # Read queries
    print("Reading workload queries...")
    with open('/workspace/data/workloads/tpch_sf1/TP_queries.sql', 'r') as f:
        tp_queries = f.readlines()
    with open('/workspace/data/workloads/tpch_sf1/workload_100k_s1_group_order_by_more_complex.sql', 'r') as f:
        ap_queries = f.readlines()
        
    # Combine workload samples
    all_queries = tp_queries[:20] + ap_queries[:10]
    
    # Run validation
    validator.validate_implementation(all_queries)

if __name__ == "__main__":
    main()