from index_benefit_estimator import QueryCostEstimator, Index, Column
from typing import List, Dict, Tuple
import re

class WorkloadSimulator:
    def __init__(self):
        self.estimator = QueryCostEstimator()
        
    def analyze_query_patterns(self, queries: List[str]) -> Dict[str, Dict]:
        """Analyze patterns in the query workload"""
        patterns = {
            'tables': {},           # Table access frequency
            'conditions': {},       # Column conditions
            'joins': {},           # Join patterns
            'columns': {}          # Column usage
        }
        
        for query in queries:
            # Extract tables
            tables = re.findall(r'FROM\s+"([^"]+)"|\bJOIN\s+"([^"]+)"', query)
            for table_match in tables:
                table = next(t for t in table_match if t)
                patterns['tables'][table] = patterns['tables'].get(table, 0) + 1
            
            # Extract conditions
            conditions = self.estimator.parse_where_conditions(query)
            for table, column, op, _ in conditions:
                key = f"{table}.{column}"
                if key not in patterns['conditions']:
                    patterns['conditions'][key] = {'ops': {}, 'count': 0}
                patterns['conditions'][key]['count'] += 1
                patterns['conditions'][key]['ops'][op] = patterns['conditions'][key]['ops'].get(op, 0) + 1
            
            # Extract joins
            joins = self.estimator.extract_joins(query)
            for t1, c1, t2, c2 in joins:
                key = f"{t1}.{c1}={t2}.{c2}"
                patterns['joins'][key] = patterns['joins'].get(key, 0) + 1
                
        return patterns
    
    def suggest_indexes(self, patterns: Dict[str, Dict]) -> List[Index]:
        """Suggest indexes based on query patterns"""
        suggested_indexes = []
        
        # Suggest indexes for frequently used conditions
        for col_key, stats in patterns['conditions'].items():
            if stats['count'] >= 2:  # Column used in multiple queries
                table, column = col_key.split('.')
                # Check if it's used in equality conditions
                has_equality = stats['ops'].get('=', 0) > 0
                suggested_indexes.append(
                    Index(table, [Column(table, column)], is_unique=has_equality)
                )
        
        # Suggest indexes for frequent joins
        for join_key, count in patterns['joins'].items():
            if count >= 2:  # Join used multiple times
                t1_c1, t2_c2 = join_key.split('=')
                t1, c1 = t1_c1.split('.')
                t2, c2 = t2_c2.split('.')
                suggested_indexes.extend([
                    Index(t1, [Column(t1, c1)]),
                    Index(t2, [Column(t2, c2)])
                ])
        
        # Deduplicate indexes
        unique_indexes = {}
        for idx in suggested_indexes:
            key = f"{idx.table}.{'.'.join(str(c) for c in idx.columns)}"
            if key not in unique_indexes or (idx.is_unique and not unique_indexes[key].is_unique):
                unique_indexes[key] = idx
        
        return list(unique_indexes.values())
    
    def evaluate_workload(self, queries: List[str], indexes: List[Index]) -> Dict[str, float]:
        """Evaluate workload cost with and without indexes"""
        total_cost_no_indexes = 0
        total_cost_with_indexes = 0
        improvements = []
        
        for i, query in enumerate(queries):
            cost_no_idx = self.estimator.estimate_query_cost(query)
            cost_with_idx = self.estimator.estimate_query_cost(query, indexes)
            
            total_cost_no_indexes += cost_no_idx
            total_cost_with_indexes += cost_with_idx
            
            if cost_no_idx > 0:
                improvement = (cost_no_idx - cost_with_idx) / cost_no_idx * 100
                improvements.append(improvement)
        
        return {
            'total_improvement': ((total_cost_no_indexes - total_cost_with_indexes) / total_cost_no_indexes * 100) 
                if total_cost_no_indexes > 0 else 0,
            'avg_improvement': sum(improvements) / len(improvements) if improvements else 0,
            'max_improvement': max(improvements) if improvements else 0,
            'min_improvement': min(improvements) if improvements else 0
        }

def main():
    simulator = WorkloadSimulator()
    
    # Read queries
    print("Reading workload queries...")
    with open('/workspace/data/workloads/tpch_sf1/TP_queries.sql', 'r') as f:
        tp_queries = f.readlines()
    with open('/workspace/data/workloads/tpch_sf1/workload_100k_s1_group_order_by_more_complex.sql', 'r') as f:
        ap_queries = f.readlines()
    
    # Analyze TP queries
    print("\nAnalyzing TP (point) queries...")
    tp_patterns = simulator.analyze_query_patterns(tp_queries[:20])  # Analyze first 20 queries
    tp_indexes = simulator.suggest_indexes(tp_patterns)
    
    print("\nSuggested indexes for TP workload:")
    for idx in tp_indexes:
        print(f"- {idx}")
    
    tp_eval = simulator.evaluate_workload(tp_queries[:20], tp_indexes)
    print("\nTP workload evaluation:")
    print(f"Total improvement: {tp_eval['total_improvement']:.1f}%")
    print(f"Average query improvement: {tp_eval['avg_improvement']:.1f}%")
    print(f"Best case improvement: {tp_eval['max_improvement']:.1f}%")
    print(f"Worst case improvement: {tp_eval['min_improvement']:.1f}%")
    
    # Analyze AP queries
    print("\nAnalyzing AP (analytical) queries...")
    ap_patterns = simulator.analyze_query_patterns(ap_queries[:10])  # Analyze first 10 queries
    ap_indexes = simulator.suggest_indexes(ap_patterns)
    
    print("\nSuggested indexes for AP workload:")
    for idx in ap_indexes:
        print(f"- {idx}")
    
    ap_eval = simulator.evaluate_workload(ap_queries[:10], ap_indexes)
    print("\nAP workload evaluation:")
    print(f"Total improvement: {ap_eval['total_improvement']:.1f}%")
    print(f"Average query improvement: {ap_eval['avg_improvement']:.1f}%")
    print(f"Best case improvement: {ap_eval['max_improvement']:.1f}%")
    print(f"Worst case improvement: {ap_eval['min_improvement']:.1f}%")
    
    # Analyze combined workload
    print("\nAnalyzing combined workload...")
    combined_indexes = list({idx for idx in tp_indexes + ap_indexes})  # Deduplicate using set
    
    print("\nIndex Analysis:")
    all_queries = tp_queries[:20] + ap_queries[:10]
    index_stats = []
    
    for idx in combined_indexes:
        # Calculate various metrics for each index
        total_benefit = 0
        queries_improved = 0
        max_improvement = 0
        total_improvement_pct = 0
        
        for query in all_queries:
            cost_no_idx = simulator.estimator.estimate_query_cost(query)
            cost_with_idx = simulator.estimator.estimate_query_cost(query, [idx])
            benefit = cost_no_idx - cost_with_idx
            
            if benefit > 0:
                queries_improved += 1
                improvement_pct = (benefit / cost_no_idx) * 100
                max_improvement = max(max_improvement, improvement_pct)
                total_improvement_pct += improvement_pct
                total_benefit += benefit
        
        avg_improvement = total_improvement_pct / queries_improved if queries_improved > 0 else 0
        benefit_score = idx.get_benefit_score() * (1 + avg_improvement/100)
        
        index_stats.append({
            'index': idx,
            'total_benefit': total_benefit,
            'queries_improved': queries_improved,
            'max_improvement': max_improvement,
            'avg_improvement': avg_improvement,
            'benefit_score': benefit_score
        })
    
    # Sort indexes by benefit score
    index_stats.sort(key=lambda x: x['benefit_score'], reverse=True)
    
    print("\nTop 10 Most Beneficial Indexes:")
    print("=" * 80)
    print(f"{'Index':<40} {'Queries':<10} {'Avg Imp':<10} {'Max Imp':<10} {'Score':<10}")
    print("-" * 80)
    
    for stat in index_stats[:10]:
        idx_str = str(stat['index'])
        if len(idx_str) > 37:
            idx_str = idx_str[:34] + "..."
        print(f"{idx_str:<40} {stat['queries_improved']:>8} {stat['avg_improvement']:>8.1f}% {stat['max_improvement']:>8.1f}% {stat['benefit_score']:>8.1f}")
    
    # Analyze index combinations
    print("\nAnalyzing index combinations...")
    top_5_indexes = [stat['index'] for stat in index_stats[:5]]
    
    base_cost = sum(simulator.estimator.estimate_query_cost(q) for q in all_queries)
    cost_with_all = sum(simulator.estimator.estimate_query_cost(q, top_5_indexes) for q in all_queries)
    total_improvement = ((base_cost - cost_with_all) / base_cost) * 100 if base_cost > 0 else 0
    
    print(f"\nCombined benefit of top 5 indexes: {total_improvement:.1f}% total cost reduction")
    
    # Storage impact analysis (rough estimation)
    print("\nEstimated Storage Impact:")
    total_rows = sum(simulator.estimator.table_sizes.values())
    for idx in top_5_indexes:
        # Rough estimation: each index entry takes about 20 bytes + column sizes
        estimated_size_mb = (simulator.estimator.table_sizes[idx.table] * 20 * len(idx.columns)) / (1024 * 1024)
        print(f"- {idx}: {estimated_size_mb:.1f} MB")

if __name__ == "__main__":
    main()