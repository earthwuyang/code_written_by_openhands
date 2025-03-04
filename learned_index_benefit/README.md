# Learned Index Benefit Estimation Model

This project implements a cost-based model for estimating query execution costs with hypothetical indexes in PolarDB-IMCI. The model helps in automatic index selection by predicting the benefits of potential indexes without actually creating them.

## Model Components

### 1. Cost Estimation
- **Table Statistics**: Uses table sizes and data distribution statistics
- **Index Costs**: Models B-tree lookup and data access costs
- **Join Costs**: Considers nested loop, hash join, and index nested loop join strategies
- **Selectivity Estimation**: Adaptive selectivity based on operators and index properties

### 2. Index Types Support
- **Single-column Indexes**: Traditional B-tree indexes on individual columns
- **Composite Indexes**: Multi-column indexes with leading column consideration
- **Primary Key Indexes**: Clustered unique indexes
- **Foreign Key Indexes**: Non-clustered indexes for join optimization

### 3. Cost Model Features

#### Base Costs
- Table Scan: Linear cost based on table size
- Index Lookup: Logarithmic cost for B-tree traversal + data access cost
- Join Operations: Considers different join strategies and index availability

#### Selectivity Factors
- Point Queries (=): 0.001 (1/1000)
- Range Queries (<, >, etc.): 0.3
- LIKE/Pattern Matching: 0.05
- List Operations (IN): 0.05
- Unique Indexes: Minimum selectivity based on table size

#### Cost Adjustments
- Random I/O penalty for non-clustered indexes
- Composite index penalties for non-leading column usage
- Join order optimization based on available indexes
- Filter pushdown consideration after joins

## Usage Example

```python
# Create an index benefit estimator
estimator = QueryCostEstimator()

# Define potential indexes
indexes = [
    Index('orders', [Column('orders', 'o_orderkey')], True),
    Index('lineitem', [Column('lineitem', 'l_orderkey')])
]

# Estimate query cost with and without indexes
base_cost = estimator.estimate_query_cost(query)
cost_with_indexes = estimator.estimate_query_cost(query, indexes)
benefit = base_cost - cost_with_indexes
```

## Experimental Results

### Point Query Performance
- Up to 100% cost reduction for point queries with matching unique indexes
- 80-99% improvement for point queries with composite indexes
- Effective identification of beneficial specialized indexes

### Analytical Query Performance
- 10-15% cost reduction for complex analytical queries
- Significant benefits from composite indexes on frequently joined tables
- Effective join order optimization with available indexes

### Composite Index Benefits
The model identifies valuable composite index opportunities:
1. lineitem table: ~6M cost reduction
2. orders table: ~1.5M cost reduction
3. part table: ~176K cost reduction
4. supplier table: ~10K cost reduction

## Implementation Details

### Cost Calculation
1. **Index Lookup Cost**:
   ```python
   lookup_cost = log2(table_size) * index_factor
   data_access_cost = rows_to_access * access_factor
   total_cost = lookup_cost + data_access_cost
   ```

2. **Join Cost**:
   ```python
   hash_join_cost = 3 * (size1 + size2)
   index_join_cost = size1 * log2(size2)
   best_cost = min(nested_loop_cost, hash_join_cost, index_join_cost)
   ```

### Selectivity Estimation
- Uses table statistics and operator characteristics
- Adjusts for index properties (unique vs. non-unique)
- Considers composite index column order

## Future Improvements

1. **Machine Learning Integration**:
   - Train models on actual query execution statistics
   - Learn workload-specific patterns
   - Adapt cost factors based on system performance

2. **Enhanced Statistics**:
   - Column value distribution
   - Correlation between columns
   - Data skewness factors

3. **Dynamic Adjustment**:
   - Runtime feedback for cost model
   - Workload-based selectivity adjustment
   - System resource consideration