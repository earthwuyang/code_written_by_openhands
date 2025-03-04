import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

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
        self.labels = []
        self._process_plans()
        
        # Convert to tensors and normalize
        self.features = torch.stack(self.features_list)
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        
        # Normalize features and labels
        self.features = (self.features - self.features.mean(dim=0)) / (self.features.std(dim=0) + 1e-6)
        self.labels = self.labels / self.labels.mean()
        
    def _process_plans(self):
        for plan_dict in self.plans:
            features = self._extract_plan_features(plan_dict['Plan'])
            self.features_list.append(features)
            self.labels.append(plan_dict['peakmem'])
    
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
    
    def _calculate_feature_size(self):
        # Calculate base feature size
        base_features = len(self._extract_node_features({}))
        # For a binary tree of depth 3, calculate total feature size
        # Level 0: 1 node, Level 1: 2 nodes, Level 2: 4 nodes
        total_nodes = 1 + 2 + 4  # Sum of nodes at each level
        return base_features * total_nodes

    def _extract_plan_features(self, plan, max_depth=2, max_children=2):
        base_feature_size = len(self._extract_node_features({}))
        total_feature_size = self._calculate_feature_size()
        
        def process_node(node, depth=0, pos=0):
            if depth >= max_depth or pos >= total_feature_size:
                return []
            
            # Get current node features
            node_features = self._extract_node_features(node)
            
            # Ensure node_features has exactly base_feature_size elements
            node_features = (node_features + [0] * base_feature_size)[:base_feature_size]
            
            # Process child plans
            all_features = node_features.copy()
            if 'Plans' in node:
                for i, child in enumerate(node['Plans'][:max_children]):
                    child_pos = pos + base_feature_size * (2**depth) * (i + 1)
                    child_features = process_node(child, depth + 1, child_pos)
                    all_features.extend(child_features)
            
            # Pad to ensure fixed size at each level
            remaining_size = total_feature_size - len(all_features)
            if remaining_size > 0:
                all_features.extend([0] * remaining_size)
            
            return all_features[:total_feature_size]
        
        # Extract hierarchical features
        plan_features = process_node(plan)
        
        # Ensure fixed size
        if len(plan_features) < total_feature_size:
            plan_features.extend([0] * (total_feature_size - len(plan_features)))
        
        return torch.tensor(plan_features[:total_feature_size], dtype=torch.float)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TreeMemoryPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Feature processing layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        
        # Final prediction layers
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Feature processing with residual connections
        identity = x
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Final prediction
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x.squeeze()

# def train_and_evaluate(train_dataset, val_dataset, test_dataset, epochs=50):
#     device = torch.device('cpu')
    
#     # Calculate input dimension from the first sample
#     input_dim = train_dataset[0].x.size(1)
    
#     # Create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=16)
#     test_loader = DataLoader(test_dataset, batch_size=16)
    
#     # Create model
#     model = TreeMemoryPredictor(input_dim).to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
#     # Normalize target values
#     train_mean = torch.mean(torch.tensor([data.y for data in train_dataset]))
#     train_std = torch.std(torch.tensor([data.y for data in train_dataset]))
    
#     def normalize_target(y):
#         return (y - train_mean) / train_std
    
#     def denormalize_target(y):
#         return y * train_std + train_mean
    
#     # Training loop
#     best_val_loss = float('inf')
#     for epoch in range(epochs):
#         # Training
#         model.train()
#         train_loss = 0
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             out = model(batch)
#             loss = criterion(out, normalize_target(batch.y))
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * batch.num_graphs
#         train_loss /= len(train_dataset)
        
#         # Validation
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 batch = batch.to(device)
#                 out = model(batch)
#                 val_loss += criterion(out, normalize_target(batch.y)).item() * batch.num_graphs
#         val_loss /= len(val_dataset)
        
#         # Learning rate scheduling
#         scheduler.step(val_loss)
        
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), 'best_model_gnn.pt')
    
#     # Test evaluation
#     model.load_state_dict(torch.load('best_model_gnn.pt'))
#     model.eval()
    
#     predictions = []
#     actuals = []
    
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = batch.to(device)
#             out = model(batch)
#             pred = denormalize_target(out)
#             predictions.extend(pred.cpu().numpy())
#             actuals.extend(batch.y.cpu().numpy())
    
#     predictions = np.array(predictions)
#     actuals = np.array(actuals)
    
#     # Calculate Q-error
#     qerrors = np.maximum(predictions/actuals, actuals/predictions)
    
#     print("\nTest Set Results:")
#     print(f'Median Q-Error: {np.median(qerrors):.4f}')
#     print(f'90th Percentile Q-Error: {np.percentile(qerrors, 90):.4f}')
#     print(f'95th Percentile Q-Error: {np.percentile(qerrors, 95):.4f}')
#     print(f'99th Percentile Q-Error: {np.percentile(qerrors, 99):.4f}')
    
#     return qerrors

def train_and_evaluate(train_dataset, val_dataset, test_dataset, epochs=50):
    device = torch.device('cpu')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Calculate input dimension from the first sample
    input_dim = train_dataset[0][0].shape[0]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = TreeMemoryPredictor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
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
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
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
            }, 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
    
    print("\nTraining completed. Loading best model for evaluation...")
    
    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test evaluation
    test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
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
    data_dir = '/home/wuy/DB/pg_mem_data/tpch_sf1'
    train_dataset = QueryPlanDataset(os.path.join(data_dir, 'train_plans.json'))
    val_dataset = QueryPlanDataset(os.path.join(data_dir, 'val_plans.json'))
    test_dataset = QueryPlanDataset(os.path.join(data_dir, 'test_plans.json'))
    
    # Train and evaluate
    qerrors = train_and_evaluate(train_dataset, val_dataset, test_dataset)