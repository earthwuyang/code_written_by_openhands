https://github.com/earthwuyang/code_written_by_openhands is a repository containing code written by an AI agent OpenHands.
OpenHands is a powerful platform for software engineering agents powered by AI: https://github.com/All-Hands-AI/OpenHands.

## 1. row_column_routing contains the code I prompt OpenHands to write a program to train model for row column routing

### The prompt I use is:
for polardb-imci, please predict whether a query will execute faster on row storage or column storage, the query_costs.csv is located in query_costs.csv, containing query_id, use_imci two columns. use_imci=0 if row is faster, use_imci=1 if column is faster.  inside row_plans are <query_id>.json containing row plan json file of query_id . please inspect the json files of row plan  to understand the query execution plan and know how to parse them. please implement deep neural network to predict the binary classification problem: whether a query run faster on row or on column. when you pip install packages, you should -i https://pypi.tuna.tsinghua.edu.cn/simple .
I have updated my data to include much more query plans, I also update query_costs.csv to include more columns, including row_time, and column_time which are the time when the query is executed on row store or column store. Can you 1. execute current train_model.py on this extended dataset? with split the data into train, val, as well as test part. and test on test dataset. 2. inspect query_costs.csv to calculate a end-to-end runtime, for example, it the prediction model outputs 0, you add row_time, if output 1, you add column_time, then you get the end-to-end time of our AI-based query router model. 3. you also need to calculate the optimal end to end time, that is, you add min(row_time, column_time). 4. you need to implement a cost threshold based query router model, that is if the query_cost inside  a row plan json file is bigger than 50000, then you add column_time, if smaller than 50000, you add row_time, this way you get the end-to-end running time of the workloads for the cost threshold method.  5. I also put column_plans in ./data, you can also try to predict the row column faster problem based on column plans, you can try to decide whether using row plan to predict or using column plan to predict is better, or use a combined of both? you can try more complex neural network to predict, for example pytorch-geometric GNN model 
The only thing I do is to write these prompt to tell the OpenHands what he should do. I use claude-sonnet-3.5 LLM. And the Agent system automatically write code to implement these functions.
The AI has written different versions of files to do this, updating and refining. train_model_extended_modified.py is the final file:
### Experimental results:
##### End-to-end Runtime Results: 
AI Model Total Time: 0.0797 seconds

Cost Threshold Total Time: 0.0862 seconds 

Optimal Total Time: 0.0790 seconds

##### Relative Performance: 
AI Model vs Optimal: 0.87% slower than optimal 

Cost Threshold vs Optimal: 9.20% slower than optimal 

## 2. memory_prediction directory contains code I prompt OpenHands to write code to predict the memory consumption of query. That's what I have done in my last paper MemQ.
### The prompt I use is:
I want to predict the peak memory consumption of the execution of a query,  the plans are shown in train_plans.json, val_plans.json, test_plans.json, including peakmem as the peak memory consumption of execution of the query. please implement PyTorch geometric GNN model to do the regression task. use train_plans.json to train, val_plans.json to validate and test_plans.json to test. finally report your Qerror on test set. you can pip install -i https://pypi.tuna.tsinghua.edu.cn/simple
please, 1. first scan all json plans to collect all operator types instead of define only several operator types as python list. 2. add more features from query plans, capturing their tree-structured graph structure and implement complex pytorch geometric GNN models
try improving the model further, please try again to install pytorch and torch-geometric and utilize GCNConv, please capture the dependencies between operators.
you don’t need to install torch-scatter and torch-sparse, simply using torch and torch-geometric is enough
please report qerror of your regression model to me
The AI agent has written various versions of files to implement this and query_plan_gnn.py is the final correctly working version.
### Experimental results:

median qerror: 1.9732

90th qerror： 6.1512

95th qerror: 7.5715

99th qerror: 16.1631

## 3. online_isp is the directory containing code I prompted OpenHands to update an existing offline index selection evaluation framework into an online index selection framework and implement MAB (Multi-arm bandit) algorithm.
The original index selection evalution framework is: https://github.com/hyrise/index_selection_evaluation
### The prompt I use is:
please understand current offline index selection framework that has already been downloaded in the current  directory by reading the files under current directory, and update it to be an online index selection evaluation framework with dynamically changing workloads. And implement multi-arm bandit algorithm to incrementally update and remove indexes due to workload change, and compare the online multi-arm bandit algorithm with periodically invoking extend algorithm for changing workloads. if you need to pip install packages, you can -i https://pypi.tuna.tsinghua.edu.cn/simple  . you have access to postgres via host: 172.17.0.1 and user wuy, password: ‘wuy’, database:indexselection_tpch___1

under current directory is an implemented online index selection evaluation framework, in run_online_benchmark_new.py, we can compare onlinemab algorithm to periodically invoked extend, db2advis, relaxation algorithm and compare their performance. However, onlinemab algorithm is not a new idea. Can you devise a novel online solution to online index selection ,and implement your algorithm under selection algorithm directory. And run the experiment to assess your new algorithm’s online performance against onlinemab, and original offline algorithms extend etc.    Your new algorithm is expected to achieve similar performance with the online mab and other offline algorithms periodically invoked for online case and use much less recommendation time. You should start by inspecting current files and understand the codebase structure, and then propose a new idea for online index selection: to incrementally update and delete indexes in response to workload change. you should also update current framework to support data update and insert and measure online index selection algorithm’s performance under data distribution change. Then you implement your idea and run experiments. finally you are expected to write a complete latex paper describing your methods and experimental results. Your experimental results are expected to be thorough and detailed , covering all aspects of your algorithm. If you need to pip install packages, use -i https://pypi.tuna.tsinghua.edu.cn/simple. you have access to postgres via host: 172.17.0.1 and user wuy, password: ‘wuy’, database:indexselection_tpch___1

### Experimental results: 
bandit is implemented by OpenHands, DB2Advis, Relaxation, Extend are baselines provided by the original offline isp framework.

(1)Cost improvement vs no-index:

bandit (openhands): 26.69%

DB2Advis: 26.53%

Relaxation: 26.44%

Extend: 26.38%

(2)Recommendation time (the time the recommendation algorithm takes to recommend indexes):

Bandit: 22.38

DB2Advis: 494.42

Relaxation: 15497.91

Extend:302.16

(3) Total cost of workload execution:

No-index: 40097463.66

Bandit: 29394255.52

DB2Advis: 29464563.18

Relaxation: 29493770.70

Extend: 29519541.84

### Conclusion:
The bandit algorithm implemented by AI has comparable performance to other baselines while only has a small recommendation time, showing its superority in online index selection.


## 4. learned_index_benefit is a failed attempt for AI to estimate the cost of query given some simulated indexes. That's to say, to mimick the function of hypopg extention of Postgres.
### The prompt I use is: 
You are expected to implement a Machine learning project to estimate the query cost given a query and some indexes are built. 
However, you should not actually create indexes, you should input the features of indexes to be built and the feautures of a query into a model and output the query cost supposing the indexes exist.
You have access to polardb-imci database via user user1, port 22224, host:172.17.0.1, password 'your_password', database: tpch_sf1. instead of installing mysql-client, you can use python pymysql to test connection.
The polardb-imci is mysql-compatible, you can access polardb-imci the same as you access a mysql database.
under data directory, there are workloads/tpch_sf1, there are two files:TP_queries.sql  workload_100k_s1_group_order_by_more_complex.sql, one is TP point queires, one is AP analytical queries, these query files should be used to train your model. 
You can read a paper entitled 'learned index benefit' to get some idea on how to estimate the query cost of a query given some supposed-to-exist indexes.
You need to do this because I want to do automatic index selection for polardb-imci database, but polar-imci does not support what-if hypothetical index utility so that I can't directly easily estimate the cost of a query supposing some indexes exist.
You may need to interact with the tpch database of polardb-icmi to generate training data for estimate the query cost under some supposed-to-exist indexes and train your machine learning model.
You can also first choose not to use machine learning model but simply using some cost functions to calculate the query cost under some indexes.
After you implement your code you should run the experiments, report experimental results (how well your model can estimate query cost compared with the indexes are actually implemented)
Finally you should document how you implement your cost model about learned index benefits , that's to say a model that can estimate the query cost under some simulated indexes so that i can calculate the benefit of creating some indexes for the execution of some workloads and finally for the ultimate goal of index selection for polardb-imci with your learned index benefit estimation model.
That's to say finally I can estimate the benefit of creating indexes without actually implementing the creation but only calculate their beneift for each query.
You can first do some literature review, for example the learned index benefit paper and other relevant research papers to get some ideas and then implement code and run experiments. when you debug, you can use indexselection_tpch___0_1 which is tpch of scale factor of 0.1 which is smaller and faster to query when you debug. 
you have previously implemented a lot of files under the directory, please first understand the files and  please run validate_cost_model.py and debug 

All files under learned_index_benefit are mainly written by AI including codes and documents.
### Experimetal results:
Detailed Query Analysis:
| Query ID | Est. Cost | Act. Cost | Error    | Issue          | Factor    | Est. (NI) | Act. (NI) |
|----------|-----------|-----------|----------|----------------|-----------|-----------|-----------|
| 1        | 1285364   | 429       | 2993.44  | overestimation | 2994.4x   | 1299997   | 429       |
| 3        | 1128571   | 122985    | 8.18     | overestimation | 9.2x      | 1128571   | 122985    |
| 9        | 1345627   | 2468391   | 0.45     | underestimation| 1.8x      | 1356075   | 2468391   |
| 13       | 1279812   | 371410    | 2.45     | overestimation | 3.4x      | 1421304   | 371410    |
| 21       | 824643    | 684       | 1204.19  | overestimation | 1205.2x   | 1128571   | 684       |
| 27       | 730311    | 866475    | 0.16     | underestimation| 1.2x      | 1145362   | 866475    |
| 36       | 930575    | 178       | 5226.66  | overestimation | 5227.7x   | 1345626   | 178       |
| 39       | 1128062   | 309       | 3651.93  | overestimation | 3652.9x   | 1128571   | 309       |
| 40       | 1021107   | 1410      | 722.94   | overestimation | 723.9x    | 1021107   | 1410      |
| 47       | 1128571   | 1093      | 1031.30  | overestimation | 1032.3x   | 1128571   | 1093      |
| 54       | 1190504   | 286046    | 3.16     | overestimation | 4.2x      | 1190504   | 286046    |
| 60       | 287896    | 126       | 2288.79  | overestimation | 2289.8x   | 287896    | 126       |
| 70       | 1020599   | 2156      | 472.48   | overestimation | 473.5x    | 1021107   | 2156      |
| 74       | 1128571   | 40657     | 26.76    | overestimation | 27.8x     | 1128571   | 40657     |
| 75       | 1057824   | 1084      | 974.57   | overestimation | 975.6x    | 1128571   | 1084      |
| 77       | 1128571   | 156       | 7232.50  | overestimation | 7233.5x   | 1128571   | 156       |
| 79       | 1057824   | 644       | 1642.17  | overestimation | 1643.2x   | 1128571   | 644       |
| 83       | 401020    | 4         | 112329.42| overestimation | 112330.4x | 401020    | 4         |
| 84       | 1050361   | 124       | 8450.57  | overestimation | 8451.6x   | 1128571   | 124       |
| 85       | 1128571   | 5452600   | 0.79     | underestimation| 4.8x      | 1128571   | 5452600   |
| 86       | 1050504   | 5751086   | 0.82     | underestimation| 5.5x      | 1050504   | 5751086   |
| 96       | 635452    | 92        | 6933.98  | overestimation | 6935.0x   | 635452    | 92        |
| 99       | 200003    | 2         | 85837.20 | overestimation | 85838.2x  | 200003    | 2         |

### Problematic Queries (Error Ratio > 2.0):
Query 83 (overestimation, Factor: 112330.4x):

Query: SELECT COUNT(*) as agg_0 FROM "nation" JOIN "region" ON "nation"."n_regionkey" = "region"."r_regionk...
Estimated Cost: 401020 Actual Cost: 4 Error Ratio: 112329.42

(No Index) Estimated Cost: 401020 Actual Cost: 4 Error Ratio: 112329.42


Query 99 (overestimation, Factor: 85838.2x):

Query: SELECT COUNT(*) as agg_0, AVG("region"."r_regionkey" + "nation"."n_nationkey") as agg_1 FROM "region...
Estimated Cost: 200003 Actual Cost: 2 Error Ratio: 85837.20

(No Index) Estimated Cost: 200003 Actual Cost: 2 Error Ratio: 85837.20

Query 84 (overestimation, Factor: 8451.6x):

Query: SELECT MIN("lineitem"."l_extendedprice" + "lineitem"."l_tax") as agg_0, AVG("lineitem"."l_extendedpr...
Estimated Cost: 1050361 Actual Cost: 124 Error Ratio: 8450.57

(No Index) Estimated Cost: 1128571 Actual Cost: 124 Error Ratio: 9079.87

### Recommendations:

● Cost model tends to overestimate significantly

● Consider reducing cost factors in estimation

● Review selectivity estimation for complex predicates

● Cost model needs significant improvement

● Consider recalibrating selectivity estimates

● Review statistics collection process

### Conclusion:
As we can see ,although AI writes a lot of code to estimate the the query cost given the mock existence of some indexes, the estimation accuracy is extremely low. The error rate is too high for practical use.


## Some progress on AI automatically writing codes (a.k.a. software engineering agents):
1. https://arxiv.org/abs/2502.18449 
SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
2. https://www.swebench.com
Can Language Models Resolve Real-World GitHub Issues?
ICLR 2024
3. https://agentlaboratory.github.io/
AI自动撰写论文，AMD开源自动完成科研全流程的多智能体框架
4. https://github.com/du-nlp-lab/MLR-Copilot
MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents
5. https://arxiv.org/abs/2408.00665
发表于CCF A类会议MM上的一篇论文：
AutoM3L: An Automated Multimodal Machine Learning Framework with Large Language Models
我希望的是在DB领域，也可以做一个利用LLM进行ML4DB自动科研的项目。
