import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# All operator types we found in the plans
OPERATOR_TYPES = [
    'Aggregate', 'Gather', 'Gather Merge', 'Hash', 'Hash Join',
    'Incremental Sort', 'Limit', 'Materialize', 'Merge Join',
    'Nested Loop', 'Result', 'Seq Scan', 'Sort'
]

class QueryPlanDataset(Dataset):
    def __init__(self, plan_file):
        super().__init__()
        with open(plan_file, 'r') as f:
            self.plans = json.load(f)
        self.features_list = []
        self.adj_list = []
        self.labels = []
        self._process_plans()
        
        # Convert to tensors and normalize
        self.features = torch.stack(self.features_list)
        self.adj_matrices = torch.stack(self.adj_list)
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        
        # Normalize features and labels
        self.features = (self.features - self.features.mean(dim=0)) / (self.features.std(dim=0) + 1e-6)
        self.labels = self.labels / self.labels.mean()
    
    def _extract_node_features(self, plan):
        # Basic numeric features
        numeric_features = [
            plan.get('Total Cost', 0),
            plan.get('Plan Rows', 0),
            plan.get('Plan Width', 0),
            plan.get('Startup Cost', 0),
            1 if plan.get('Parallel Aware', False) else 0,
            1 if plan.get('Async Capable', False) else 0
        ]
        
        # Node type one-hot encoding
        node_type = plan.get('Node Type', '')
        node_type_onehot = [1 if node_type == op_type else 0 for op_type in OPERATOR_TYPES]
        
        # Additional features
        join_type = plan.get('Join Type', '')
        join_types = ['Inner', 'Left', 'Right', 'Full', 'Semi', 'Anti']
        join_type_onehot = [1 if join_type == jt else 0 for jt in join_types]
        
        # Strategy for Aggregate/Sort
        strategy = plan.get('Strategy', '')
        strategy_types = ['Plain', 'Sorted', 'Hashed', 'Mixed']
        strategy_onehot = [1 if strategy == st else 0 for st in strategy_types]
        
        # Combine all features
        return numeric_features + node_type_onehot + join_type_onehot + strategy_onehot
    
    def _process_plans(self):
        max_nodes = 16  # Maximum number of nodes in the plan tree
        feature_dim = len(self._extract_node_features({}))
        
        for plan_dict in self.plans:
            nodes = []  # List of node features
            adj_matrix = torch.zeros((max_nodes, max_nodes))  # Adjacency matrix
            node_count = [0]  # Use list to allow modification in nested function
            
            def process_node(plan, parent_idx=None):
                if node_count[0] >= max_nodes:
                    return
                
                current_idx = node_count[0]
                nodes.append(self._extract_node_features(plan))
                node_count[0] += 1
                
                if parent_idx is not None:
                    # Add bidirectional edges
                    adj_matrix[parent_idx, current_idx] = 1
                    adj_matrix[current_idx, parent_idx] = 1
                
                if 'Plans' in plan:
                    for child_plan in plan['Plans']:
                        process_node(child_plan, current_idx)
            
            # Process the plan tree
            process_node(plan_dict['Plan'])
            
            # Pad node features if necessary
            while len(nodes) < max_nodes:
                nodes.append([0] * feature_dim)
            
            # Convert to tensors
            node_features = torch.tensor(nodes, dtype=torch.float)
            
            self.features_list.append(node_features)
            self.adj_list.append(adj_matrix)
            self.labels.append(plan_dict['peakmem'])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.adj_matrices[idx], self.labels[idx]

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        # x: batch_size x num_nodes x in_features
        # adj: batch_size x num_nodes x num_nodes
        support = torch.matmul(x, self.weight)  # batch_size x num_nodes x out_features
        output = torch.matmul(adj, support)  # batch_size x num_nodes x out_features
        return output + self.bias

class ImprovedGNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        # Graph convolution layers
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim // 2)
        self.gc3 = GraphConvolution(hidden_dim // 2, hidden_dim // 4)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Global pooling and prediction layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.fc2 = nn.Linear(hidden_dim // 8, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, adj):
        # Graph convolution layers with residual connections
        x1 = F.relu(self.gc1(x, adj))
        x1 = x1.transpose(1, 2)
        x1 = self.bn1(x1)
        x1 = x1.transpose(1, 2)
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.gc2(x1, adj))
        x2 = x2.transpose(1, 2)
        x2 = self.bn2(x2)
        x2 = x2.transpose(1, 2)
        x2 = self.dropout(x2)
        
        x3 = F.relu(self.gc3(x2, adj))
        x3 = x3.transpose(1, 2)
        x3 = self.bn3(x3)
        x3 = x3.transpose(1, 2)
        x3 = self.dropout(x3)
        
        # Global pooling
        x = x3.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        # Final prediction
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze()

def train_and_evaluate(train_dataset, val_dataset, test_dataset, epochs=50):
    device = torch.device('cpu')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Calculate input dimension from the first sample
    input_dim = train_dataset[0][0].shape[-1]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = ImprovedGNNPredictor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for features, adj, labels in train_loader:
            features, adj, labels = features.to(device), adj.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features, adj)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= len(train_loader.dataset)
            
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_actuals = []
        
        with torch.no_grad():
            for features, adj, labels in val_loader:
                features, adj, labels = features.to(device), adj.to(device), labels.to(device)
                outputs = model(features, adj)
                val_loss += criterion(outputs, labels).item() * features.size(0)
                
                # Store predictions and actuals for Q-error calculation
                val_predictions.extend(outputs.cpu().numpy())
                val_actuals.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        # Calculate validation Q-error
        val_predictions = np.array(val_predictions) * train_dataset.labels.mean().item()
        val_actuals = np.array(val_actuals) * train_dataset.labels.mean().item()
        val_qerrors = np.maximum(val_predictions/val_actuals, val_actuals/val_predictions)
        val_median_qerror = np.median(val_qerrors)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Median Q-Error: {val_median_qerror:.4f}')
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model_gnn.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
    
    print("\nTraining completed. Loading best model for evaluation...")
    
    # Load best model
    checkpoint = torch.load('best_model_gnn.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test evaluation
    test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, adj, labels in test_loader:
            features, adj, labels = features.to(device), adj.to(device), labels.to(device)
            outputs = model(features, adj)
            test_loss += criterion(outputs, labels).item() * features.size(0)
            
            # Denormalize predictions and labels
            pred = outputs * train_dataset.labels.mean()
            actual = labels * train_dataset.labels.mean()
            predictions.extend(pred.cpu().numpy())
            actuals.extend(actual.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate Q-error
    qerrors = np.maximum(predictions/actuals, actuals/predictions)
    
    print("\nTest Set Results:")
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Median Q-Error: {np.median(qerrors):.4f}')
    print(f'90th Percentile Q-Error: {np.percentile(qerrors, 90):.4f}')
    print(f'95th Percentile Q-Error: {np.percentile(qerrors, 95):.4f}')
    print(f'99th Percentile Q-Error: {np.percentile(qerrors, 99):.4f}')
    
    return qerrors

if __name__ == '__main__':
    # Create datasets
    train_dataset = QueryPlanDataset('train_plans.json')
    val_dataset = QueryPlanDataset('val_plans.json')
    test_dataset = QueryPlanDataset('test_plans.json')
    
    # Train and evaluate
    qerrors = train_and_evaluate(train_dataset, val_dataset, test_dataset)