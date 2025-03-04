import logging
import numpy as np
from typing import List, Set, Dict
from selection.selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from selection.index import Index
from selection.workload import Query, Workload
from selection.utils import b_to_mb, mb_to_b

DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "exploration_factor": 0.4,  # Even higher exploration rate (40%)
    "window_size": 20,  # Even smaller window for faster adaptation
    "min_utility_threshold": 0.00001,  # Even more permissive threshold
}

class OnlineBanditAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS
        )
        self.budget = mb_to_b(self.parameters["budget_MB"])
        self.max_index_width = self.parameters["max_index_width"]
        self.exploration_factor = self.parameters["exploration_factor"]
        self.window_size = self.parameters["window_size"]
        self.min_utility_threshold = self.parameters["min_utility_threshold"]
        
        # Initialize index statistics
        self.index_stats: Dict[Index, dict] = {}  # Stats for each index
        self.current_indexes: Set[Index] = set()  # Currently selected indexes
        self.query_window: List[Query] = []  # Sliding window of recent queries
        self.total_cost_without_indexes = 0  # Cost baseline without any indexes

    def _calculate_best_indexes(self, workload):
        """Initial index selection based on the first batch of queries"""
        logging.info("Starting online index selection with multi-arm bandit")
        
        # Initialize with first query batch
        initial_queries = workload.queries[:self.window_size]
        self.query_window = initial_queries
        
        # Calculate baseline cost without indexes
        self.total_cost_without_indexes = self.cost_evaluation.calculate_cost(
            Workload(initial_queries), [], store_size=True
        )

        # Generate initial candidate indexes
        candidates = self._generate_candidate_indexes(initial_queries)
        
        # Initialize index statistics
        for index in candidates:
            self._initialize_index_stats(index)
            
        # Select initial indexes using epsilon-greedy strategy
        self._select_indexes(candidates)
        
        return list(self.current_indexes)

    def _initialize_index_stats(self, index: Index):
        """Initialize statistics for a new index"""
        self.index_stats[index] = {
            'times_used': 0,
            'total_benefit': 0,
            'avg_benefit': 0,
            'ucb_score': float('inf')  # Start with high UCB for exploration
        }

    def _generate_candidate_indexes(self, queries: List[Query]) -> Set[Index]:
        """Generate candidate indexes from the current query window"""
        candidates = set()
        
        # Group columns by table
        table_columns = {}
        for query in queries:
            logging.debug(f"\nAnalyzing query {query.nr}:")
            logging.debug(f"  Text: {query.text}")
            logging.debug(f"  Columns: {[str(c) for c in query.columns]}")
            for column in query.columns:
                if hasattr(column, 'table') and column.table:
                    table_name = column.table.name
                    if table_name not in table_columns:
                        table_columns[table_name] = set()
                    table_columns[table_name].add(column)
        
        logging.debug("\nColumns by table:")
        for table, columns in table_columns.items():
            logging.debug(f"  {table}: {[str(c) for c in columns]}")
        
        # Generate single-column indexes for each table
        for columns in table_columns.values():
            for column in columns:
                candidates.add(Index([column]))
        
        # Generate multi-column indexes up to max_index_width
        if self.max_index_width > 1:
            base_indexes = candidates.copy()
            for index in base_indexes:
                table_name = index.table().name
                # Only combine with columns from the same table
                for column in table_columns.get(table_name, []):
                    if len(index.columns) < self.max_index_width and column not in index.columns:
                        new_index = Index(list(index.columns) + [column])
                        candidates.add(new_index)
        
        logging.debug("\nGenerated candidate indexes:")
        for idx in sorted(candidates):
            logging.debug(f"  {idx}")
        
        return candidates

    def _select_indexes(self, candidates: Set[Index]):
        """Select indexes using epsilon-greedy strategy"""
        total_size = sum(index.estimated_size for index in self.current_indexes)
        
        # Evaluate each candidate index
        for index in candidates:
            if index in self.current_indexes:
                continue
                
            # Skip if adding this index would exceed budget
            if index.estimated_size is None:
                self.cost_evaluation.estimate_size(index)
            if total_size + index.estimated_size > self.budget:
                continue
                
            # Epsilon-greedy selection
            if np.random.random() < self.exploration_factor:
                # Exploration: try new index
                self._try_add_index(index)
            else:
                # Exploitation: add index only if it has good utility
                if self._get_index_utility(index) > self.min_utility_threshold:
                    self._try_add_index(index)

    def _try_add_index(self, index: Index):
        """Try to add an index and update its statistics"""
        # Calculate cost with and without this index
        # Use only the most recent queries for faster evaluation
        recent_queries = self.query_window[-5:]  # Only use last 5 queries
        workload = Workload(recent_queries)
        
        cost_before = self.cost_evaluation.calculate_cost(
            workload, list(self.current_indexes), store_size=True
        )
        
        test_indexes = self.current_indexes | {index}
        cost_after = self.cost_evaluation.calculate_cost(
            workload, list(test_indexes), store_size=True
        )
        
        benefit = cost_before - cost_after
        
        # Update index statistics
        stats = self.index_stats[index]
        stats['times_used'] += 1
        stats['total_benefit'] += benefit
        stats['avg_benefit'] = stats['total_benefit'] / stats['times_used']
        
        # Calculate metrics
        relative_improvement = (cost_before - cost_after) / cost_before if cost_before > 0 else 0
        size_mb = b_to_mb(index.estimated_size) if index.estimated_size else 0
        benefit_per_mb = benefit / size_mb if size_mb > 0 else 0
        
        # Log detailed analysis
        logging.info(f"\nAnalyzing index {index}:")
        logging.info(f"  Table: {index.table()}")
        logging.info(f"  Columns: {[str(c) for c in index.columns]}")
        logging.info(f"  Size: {size_mb:.2f}MB")
        logging.info(f"  Cost before: {cost_before:.2f}")
        logging.info(f"  Cost after: {cost_after:.2f}")
        logging.info(f"  Absolute benefit: {benefit:.2f}")
        logging.info(f"  Relative improvement: {relative_improvement:.3%}")
        logging.info(f"  Benefit per MB: {benefit_per_mb:.2f}")
        logging.info(f"  Times used: {stats['times_used']}")
        logging.info(f"  Avg benefit: {stats['avg_benefit']:.2f}")
        
        # Add index if it provides any benefit or minimal degradation
        if cost_after <= cost_before * 1.005:  # Allow up to 0.5% degradation
            # Check if we have budget
            total_size = sum(index.estimated_size for index in self.current_indexes)
            if total_size + index.estimated_size <= self.budget:
                logging.info(f"Adding beneficial index: {index} (improvement: {relative_improvement:.3%})")
                self.current_indexes.add(index)
            else:
                # If budget exceeded, try to replace a less useful index
                if self.current_indexes:
                    worst_index = min(self.current_indexes, key=lambda x: self.index_stats[x]['avg_benefit'])
                    if self.index_stats[worst_index]['avg_benefit'] < benefit:
                        self.current_indexes.remove(worst_index)
                        self.current_indexes.add(index)
                        logging.info(f"Replaced less useful index {worst_index} with {index}")
                    else:
                        logging.info(f"Skipping index: no worse indexes to replace")
                else:
                    logging.info(f"Skipping index: budget exceeded")
        else:
            logging.info(f"Skipping non-beneficial index: {index} (no improvement)")

    def _get_index_utility(self, index: Index) -> float:
        """Calculate the utility score of an index using UCB1 formula"""
        stats = self.index_stats[index]
        if stats['times_used'] == 0:
            return float('inf')
        
        # UCB1 formula: avg_reward + sqrt(2 * ln(total_time) / times_played)
        total_time = sum(s['times_used'] for s in self.index_stats.values())
        exploration_term = np.sqrt(2 * np.log(total_time) / stats['times_used'])
        
        ucb_score = stats['avg_benefit'] + exploration_term
        stats['ucb_score'] = ucb_score
        
        return ucb_score

    def _remove_unused_indexes(self):
        """Remove indexes with low utility"""
        indexes_to_remove = set()
        
        for index in self.current_indexes:
            if self._get_index_utility(index) < self.min_utility_threshold:
                indexes_to_remove.add(index)
        
        self.current_indexes -= indexes_to_remove

    def update_workload(self, new_query: Query):
        """Update the algorithm with a new query"""
        # Update sliding window
        self.query_window.append(new_query)
        if len(self.query_window) > self.window_size:
            self.query_window.pop(0)
            
        # Generate new candidate indexes from the query
        new_candidates = self._generate_candidate_indexes([new_query])
        
        # Initialize statistics for new candidates
        for index in new_candidates:
            if index not in self.index_stats:
                self._initialize_index_stats(index)
        
        # Update index selection
        self._select_indexes(new_candidates)
        
        # Periodically remove unused indexes
        self._remove_unused_indexes()
        
        return list(self.current_indexes)