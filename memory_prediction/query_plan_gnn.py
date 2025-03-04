import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import math

# Load and preprocess data
def load_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def collect_operator_types(plans):
    operator_types = set()
    def traverse_plan(node):
        if not isinstance(node, dict):
            print(f"Warning: node is not a dict: {node}")
            return
        if 'Node Type' not in node:
            print(f"Warning: Node Type not found in node: {node.keys()}")
            return
        operator_types.add(node['Node Type'])
        if 'Plans' in node:
            for child in node['Plans']:
                traverse_plan(child)
    
    print("Collecting operator types...")
    for i, plan in enumerate(plans):
        if i % 100 == 0:
            print(f"Processing plan {i}/{len(plans)}")
        if 'Plan' not in plan:
            print(f"Warning: Plan not found in plan {i}")
            continue
        traverse_plan(plan['Plan'])
    print(f"Found {len(operator_types)} unique operator types")
    return sorted(list(operator_types))

def extract_node_features(node, operator_encoder):
    try:
        # Basic features
        op_type = operator_encoder.transform([node['Node Type']])[0]
        
        # Extract numerical features with safe conversion
        def safe_float(value):
            try:
                return float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                return 0.0
        
        features = [
            float(op_type),  # operator type as float
            safe_float(node.get('Startup Cost')),
            safe_float(node.get('Total Cost')),
            safe_float(node.get('Plan Rows')),
            safe_float(node.get('Plan Width'))
        ]
        
        # Additional features
        features.extend([
            1.0 if 'Join Type' in node else 0.0,  # Has join
            1.0 if 'Filter' in node or 'Index Cond' in node else 0.0,  # Has condition
            1.0 if node.get('Parallel Aware', False) else 0.0,  # Is parallel aware
            1.0 if node.get('Async Capable', False) else 0.0,  # Is async capable
        ])
        
        return features
    except Exception as e:
        print(f"Error extracting features from node: {e}")
        print(f"Node content: {node}")
        raise

class QueryPlanDataset(Dataset):
    def __init__(self, plans, operator_encoder):
        print(f"Initializing QueryPlanDataset with {len(plans)} plans")
        self.plans = plans
        self.operator_encoder = operator_encoder
        self.processed_plans = []
        
        for i, plan in enumerate(plans):
            if i % 100 == 0:
                print(f"Processing plan {i}/{len(plans)}")
            try:
                processed_plan = self.process_plan(plan)
                self.processed_plans.append(processed_plan)
            except Exception as e:
                print(f"Error processing plan {i}: {e}")
                print(f"Plan content: {plan}")
                raise
        
        print(f"Successfully processed {len(self.processed_plans)} plans")
    
    def process_plan(self, plan):
        if 'Plan' not in plan:
            raise ValueError("No 'Plan' key found in plan")
        if 'peakmem' not in plan:
            raise ValueError("No 'peakmem' key found in plan")
        
        nodes = []
        adj_lists = []
        node_map = {}
        current_node = 0
        
        def traverse_plan(node, parent_idx=None):
            nonlocal current_node
            if not isinstance(node, dict):
                raise ValueError(f"Node is not a dict: {node}")
            
            node_idx = current_node
            node_map[id(node)] = node_idx
            
            # Extract node features
            features = extract_node_features(node, self.operator_encoder)
            nodes.append(features)
            
            # Initialize adjacency list for this node
            while len(adj_lists) <= node_idx:
                adj_lists.append([])
            
            # Add edge from parent if exists
            if parent_idx is not None:
                adj_lists[parent_idx].append(node_idx)
                adj_lists[node_idx].append(parent_idx)
            
            current_node += 1
            
            # Process child nodes
            if 'Plans' in node:
                for child in node['Plans']:
                    traverse_plan(child, node_idx)
        
        try:
            traverse_plan(plan['Plan'])
        except Exception as e:
            print(f"Error in traverse_plan: {e}")
            raise
        
        if not nodes:
            raise ValueError("No nodes found in plan")
        
        # Convert to tensors
        try:
            max_neighbors = max(len(adj) for adj in adj_lists)
            padded_adj_lists = []
            for adj in adj_lists:
                if len(adj) < max_neighbors:
                    adj.extend([-1] * (max_neighbors - len(adj)))
                padded_adj_lists.append(adj)
            
            return {
                'nodes': torch.tensor(nodes, dtype=torch.float),
                'adj_lists': torch.tensor(padded_adj_lists, dtype=torch.long),
                'target': torch.tensor(math.log(plan['peakmem']) if plan['peakmem'] > 0 else 0, dtype=torch.float)
            }
        except Exception as e:
            print(f"Error creating tensors: {e}")
            print(f"nodes shape: {len(nodes)}x{len(nodes[0]) if nodes else 0}")
            print(f"adj_lists shape: {len(adj_lists)}x{len(adj_lists[0]) if adj_lists else 0}")
            raise
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return self.processed_plans[idx]

class QueryPlanGNN(torch.nn.Module):
    def __init__(self, num_features):
        super(QueryPlanGNN, self).__init__()
        self.hidden_size = 64
        
        # Node feature processing
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_features, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        # Message passing layers
        self.message_layer1 = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.message_layer2 = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.message_layer3 = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # Output layers
        self.output_layer1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer2 = torch.nn.Linear(self.hidden_size, 1)
        
    def message_passing(self, node_features, adj_lists):
        batch_size = node_features.size(0)
        num_nodes = node_features.size(1)
        
        # Initialize hidden states
        hidden = self.node_encoder(node_features)
        
        # Three rounds of message passing
        for message_layer in [self.message_layer1, self.message_layer2, self.message_layer3]:
            # Prepare messages
            messages = torch.zeros_like(hidden)
            
            # For each node, aggregate messages from neighbors
            for b in range(batch_size):
                for i in range(num_nodes):
                    # Get neighbors
                    neighbors = adj_lists[b, i]
                    valid_neighbors = neighbors[neighbors >= 0]
                    
                    if len(valid_neighbors) > 0:
                        # Aggregate messages from valid neighbors
                        neighbor_features = hidden[b, valid_neighbors]
                        node_feature = hidden[b, i].unsqueeze(0).expand(len(valid_neighbors), -1)
                        combined = torch.cat([node_feature, neighbor_features], dim=1)
                        message = message_layer(combined)
                        messages[b, i] = torch.mean(message, dim=0)
            
            # Update hidden states
            hidden = F.relu(hidden + messages)
            hidden = F.dropout(hidden, p=0.2, training=self.training)
        
        return hidden

    def forward(self, batch):
        # Extract batch components
        node_features = batch['nodes']
        adj_lists = batch['adj_lists']
        
        # Message passing
        node_embeddings = self.message_passing(node_features, adj_lists)
        
        # Global pooling (mean of all node embeddings)
        graph_embedding = torch.mean(node_embeddings, dim=1)
        
        # Final prediction
        x = self.output_layer1(graph_embedding)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.output_layer2(x)
        
        return x

def q_error(pred, target):
    pred = torch.exp(pred)
    target = torch.exp(target)
    return torch.max(pred/target, target/pred)

def collate_batch(batch):
    # Find maximum number of nodes and neighbors in the batch
    max_nodes = max(b['nodes'].size(0) for b in batch)
    max_neighbors = max(b['adj_lists'].size(1) for b in batch)
    
    # Pad node features and adjacency lists
    padded_nodes = []
    padded_adj_lists = []
    targets = []
    
    for data in batch:
        num_nodes = data['nodes'].size(0)
        num_neighbors = data['adj_lists'].size(1)
        
        # Pad node features
        padding = torch.zeros(max_nodes - num_nodes, data['nodes'].size(1))
        padded_nodes.append(torch.cat([data['nodes'], padding], dim=0))
        
        # Pad adjacency lists both in nodes and neighbors dimensions
        current_adj = data['adj_lists']
        if num_neighbors < max_neighbors:
            padding_neighbors = torch.full((num_nodes, max_neighbors - num_neighbors), -1, dtype=torch.long)
            current_adj = torch.cat([current_adj, padding_neighbors], dim=1)
        
        if num_nodes < max_nodes:
            padding_nodes = torch.full((max_nodes - num_nodes, max_neighbors), -1, dtype=torch.long)
            current_adj = torch.cat([current_adj, padding_nodes], dim=0)
        
        padded_adj_lists.append(current_adj)
        targets.append(data['target'])
    
    return {
        'nodes': torch.stack(padded_nodes),
        'adj_lists': torch.stack(padded_adj_lists),
        'target': torch.stack(targets)
    }

def main():
    try:
        print("Loading data files...")
        data_dir = '/home/wuy/DB/pg_mem_data/tpch_sf1'
        import os
        train_plans = load_json_file(os.path.join(data_dir, 'train_plans.json'))
        val_plans = load_json_file(os.path.join(data_dir, 'val_plans.json'))
        test_plans = load_json_file(os.path.join(data_dir, 'test_plans.json'))
        print(f"Loaded {len(train_plans)} training plans, {len(val_plans)} validation plans, {len(test_plans)} test plans")
        
        # Collect all operator types
        print("\nCollecting operator types from all plans...")
        all_operators = collect_operator_types(train_plans + val_plans + test_plans)
        print(f"Found operator types: {all_operators}")
        
        print("\nFitting operator encoder...")
        operator_encoder = LabelEncoder()
        operator_encoder.fit(all_operators)
        print(f"Number of unique operators: {len(operator_encoder.classes_)}")
    
        print("\nCreating datasets...")
        print("Creating training dataset...")
        train_dataset = QueryPlanDataset(train_plans, operator_encoder)
        print("Creating validation dataset...")
        val_dataset = QueryPlanDataset(val_plans, operator_encoder)
        print("Creating test dataset...")
        test_dataset = QueryPlanDataset(test_plans, operator_encoder)
        
        print("\nCreating data loaders...")
        batch_size = 128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)
        
        print("\nInitializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        num_features = len(train_dataset[0]['nodes'][0])
        print(f"Number of input features: {num_features}")
        
        model = QueryPlanGNN(num_features=num_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print("\nStarting training...")
        best_val_error = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(200):
            # Training
            model.train()
            total_loss = 0
            num_batches = len(train_loader)
            
            print(f"\nEpoch {epoch+1}/200:")
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 10 == 0:
                    print(f"Training batch {batch_idx}/{num_batches}", end='\r')
                
                try:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    optimizer.zero_grad()
                    pred = model(batch)
                    loss = F.mse_loss(pred.squeeze(), batch['target'])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                except Exception as e:
                    print(f"\nError in training batch {batch_idx}: {e}")
                    print(f"Batch content: {batch}")
                    raise
            
            # Validation
            model.eval()
            val_errors = []
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        pred = model(batch)
                        val_errors.extend(q_error(pred, batch['target'].view(-1, 1)).cpu().numpy())
                    except Exception as e:
                        print(f"\nError in validation: {e}")
                        print(f"Batch content: {batch}")
                        raise
            
            val_error = np.median(val_errors)
            print(f'\nEpoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, '
                  f'Validation Median Q-Error: {val_error:.4f}')
            
            # Early stopping
            if val_error < best_val_error:
                best_val_error = val_error
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"New best validation error: {val_error:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break
        
        print("\nTraining completed. Loading best model for testing...")
        model.load_state_dict(torch.load('best_model.pt'))
        model.eval()
        test_errors = []
        
        print("Evaluating on test set...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print(f"Testing batch {batch_idx}/{len(test_loader)}", end='\r')
                try:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pred = model(batch)
                    test_errors.extend(q_error(pred, batch['target'].view(-1, 1)).cpu().numpy())
                except Exception as e:
                    print(f"\nError in test batch {batch_idx}: {e}")
                    print(f"Batch content: {batch}")
                    raise
        
        print('\nTest Results:')
        print(f'Median Q-Error: {np.median(test_errors):.4f}')
        print(f'90th Percentile Q-Error: {np.percentile(test_errors, 90):.4f}')
        print(f'95th Percentile Q-Error: {np.percentile(test_errors, 95):.4f}')
        print(f'99th Percentile Q-Error: {np.percentile(test_errors, 99):.4f}')
        
    except Exception as e:
        print(f"\nError in main: {e}")
        raise

if __name__ == '__main__':
    main()