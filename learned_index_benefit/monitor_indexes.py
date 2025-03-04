from index_benefit_estimator import QueryCostEstimator, Index, Column
from typing import List, Dict, Tuple
import json
import time
from datetime import datetime
import os

class IndexMonitor:
    def __init__(self):
        self.estimator = QueryCostEstimator()
        self.history_file = 'index_performance_history.json'
        self.recommended_indexes = [
            Index('orders', [Column('orders', 'o_totalprice')], True),
            Index('part', [Column('part', 'p_container')], False),
            Index('lineitem', [Column('lineitem', 'l_suppkey')], False),
            Index('part', [Column('part', 'p_size')], False),
            Index('customer', [Column('customer', 'c_custkey')], True)
        ]
        
    def load_history(self) -> List[Dict]:
        """Load historical performance data"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
        
    def save_history(self, data: List[Dict]):
        """Save performance data to history"""
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def analyze_workload(self, queries: List[str]) -> Dict:
        """Analyze current workload performance"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(queries),
            'index_performance': [],
            'storage_impact': {},
            'overall_improvement': 0
        }
        
        # Analyze each index
        for idx in self.recommended_indexes:
            perf = self.analyze_index_performance(queries, idx)
            results['index_performance'].append({
                'index': str(idx),
                'queries_improved': perf['queries_improved'],
                'avg_improvement': perf['avg_improvement'],
                'max_improvement': perf['max_improvement']
            })
            
            # Calculate storage impact
            table_size = self.estimator.table_sizes.get(idx.table, 0)
            storage_mb = (table_size * 20 * len(idx.columns)) / (1024 * 1024)
            results['storage_impact'][str(idx)] = storage_mb
            
        # Calculate overall workload improvement
        base_cost = sum(self.estimator.estimate_query_cost(q) for q in queries)
        cost_with_all = sum(self.estimator.estimate_query_cost(q, self.recommended_indexes) for q in queries)
        if base_cost > 0:
            results['overall_improvement'] = ((base_cost - cost_with_all) / base_cost) * 100
            
        return results
        
    def analyze_index_performance(self, queries: List[str], index: Index) -> Dict:
        """Analyze performance impact of a single index"""
        queries_improved = 0
        total_improvement = 0
        max_improvement = 0
        
        for query in queries:
            base_cost = self.estimator.estimate_query_cost(query)
            cost_with_index = self.estimator.estimate_query_cost(query, [index])
            
            if cost_with_index < base_cost:
                queries_improved += 1
                improvement = ((base_cost - cost_with_index) / base_cost) * 100
                total_improvement += improvement
                max_improvement = max(max_improvement, improvement)
                
        return {
            'queries_improved': queries_improved,
            'avg_improvement': total_improvement / queries_improved if queries_improved > 0 else 0,
            'max_improvement': max_improvement
        }
        
    def generate_report(self, current_results: Dict, history: List[Dict] = None):
        """Generate a monitoring report"""
        print("\nIndex Performance Monitoring Report")
        print("=" * 80)
        print(f"Timestamp: {current_results['timestamp']}")
        print(f"Total Queries Analyzed: {current_results['total_queries']}")
        print(f"Overall Workload Improvement: {current_results['overall_improvement']:.1f}%")
        
        print("\nIndex Performance:")
        print("-" * 80)
        print(f"{'Index':<40} {'Queries':<10} {'Avg Imp':<10} {'Max Imp':<10} {'Size (MB)':<10}")
        print("-" * 80)
        
        for perf in current_results['index_performance']:
            idx_str = perf['index']
            if len(idx_str) > 37:
                idx_str = idx_str[:34] + "..."
            storage = current_results['storage_impact'].get(perf['index'], 0)
            print(f"{idx_str:<40} {perf['queries_improved']:>8} {perf['avg_improvement']:>8.1f}% {perf['max_improvement']:>8.1f}% {storage:>8.1f}")
            
        if history:
            print("\nTrend Analysis:")
            print("-" * 80)
            prev_result = history[-1]
            improvement_change = current_results['overall_improvement'] - prev_result['overall_improvement']
            print(f"Performance change since last analysis: {improvement_change:+.1f}%")
            
            # Alert on significant changes
            for curr_perf in current_results['index_performance']:
                prev_perf = next((p for p in prev_result['index_performance'] if p['index'] == curr_perf['index']), None)
                if prev_perf:
                    change = curr_perf['queries_improved'] - prev_perf['queries_improved']
                    if abs(change) >= 2:
                        print(f"\nAlert: {curr_perf['index']} shows significant change in usage ({change:+d} queries)")

def main():
    monitor = IndexMonitor()
    
    # Read queries
    print("Reading workload queries...")
    with open('/workspace/data/workloads/tpch_sf1/TP_queries.sql', 'r') as f:
        tp_queries = f.readlines()
    with open('/workspace/data/workloads/tpch_sf1/workload_100k_s1_group_order_by_more_complex.sql', 'r') as f:
        ap_queries = f.readlines()
        
    # Combine workload samples
    all_queries = tp_queries[:20] + ap_queries[:10]
    
    # Load history and analyze current workload
    history = monitor.load_history()
    current_results = monitor.analyze_workload(all_queries)
    
    # Generate report
    monitor.generate_report(current_results, history)
    
    # Save results to history
    history.append(current_results)
    monitor.save_history(history)

if __name__ == "__main__":
    main()