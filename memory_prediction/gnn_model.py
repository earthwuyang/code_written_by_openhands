import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class QueryPlanDataset(Dataset):
    def __init__(self, plan_file, mem_file=None):
        # Load plans
        with open(plan_file, 'r') as f:
            self.plans = json.load(f)
            
        # Process plans to extract features
        self.features = []
        self.labels = []
        
        for plan in self.plans:
            # Extract peak memory
            peak_mem = plan['peakmem']
            self.labels.append(peak_mem)
            
            # Extract plan features
            features = self._extract_features(plan['Plan'])
            self.features.append(features)
            
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(self.labels)
        
        # Normalize features and labels
        self.features = (self.features - self.features.mean(dim=0)) / (self.features.std(dim=0) + 1e-6)
        self.labels = self.labels / self.labels.mean()  # Scale labels relative to mean
        
    def _extract_features(self, plan):
        # Basic features from the plan
        features = [
            plan.get('Total Cost', 0),
            plan.get('Plan Rows', 0),
            plan.get('Plan Width', 0),
            plan.get('Startup Cost', 0),
            1 if plan.get('Parallel Aware', False) else 0,
            1 if plan.get('Async Capable', False) else 0
        ]
        
        # Node type one-hot encoding
        node_types = ['Seq Scan', 'Hash', 'Hash Join', 'Aggregate', 'Sort', 'Nested Loop']
        node_type = plan.get('Node Type', '')
        for nt in node_types:
            features.append(1 if node_type == nt else 0)
            
        # Recursively process child plans
        if 'Plans' in plan:
            for child_plan in plan['Plans']:
                child_features = self._extract_features(child_plan)
                features.extend(child_features)
        
        # Pad or truncate to fixed length
        max_features = 100
        if len(features) > max_features:
            features = features[:max_features]
        else:
            features.extend([0] * (max_features - len(features)))
            
        return features
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MemoryPredictor(nn.Module):
    def __init__(self, input_size=100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.layers(x).squeeze()

def train_and_evaluate(train_dataset, val_dataset, test_dataset, epochs=50):
    device = torch.device('cpu')  # Use CPU
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Create model
    model = MemoryPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_loss = float('inf')
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
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Test evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            # Denormalize predictions and labels
            pred = outputs * train_dataset.labels.mean()
            actual = labels * train_dataset.labels.mean()
            predictions.extend(pred.cpu().numpy())
            actuals.extend(actual.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate Q-error
    qerrors = np.maximum(predictions/actuals, actuals/predictions)
    
    print("\nTest Set Results:")
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