import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def safe_float(value):
    """Convert string to float, handling different number formats"""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        # Replace comma with dot and try to convert
        return float(str(value).replace(',', '.'))
    except (ValueError, TypeError):
        return 0.0

def extract_features_from_plan(plan):
    features = {
        'impossible_where': 0,
        'total_cost': 0.0,
        'table_count': 0,
        'total_rows_examined': 0,
        'total_read_cost': 0,
        'total_eval_cost': 0,
        'range_access': 0,
        'ref_access': 0,
        'eq_ref_access': 0,
        'all_access': 0,
        'filesort_count': 0,
        'temp_table_count': 0,
        'total_key_length': 0,
        'total_filtered_percent': 0.0,
        'total_prefix_cost': 0.0
    }
    
    # Check for impossible WHERE clause
    if 'message' in plan['query_block'] and plan['query_block']['message'] == 'Impossible WHERE':
        features['impossible_where'] = 1
        return list(features.values())
    
    # Get query cost if available
    if 'cost_info' in plan['query_block']:
        features['total_cost'] = safe_float(plan['query_block']['cost_info']['query_cost'])
    
    def traverse_plan(node):
        nonlocal features
        if isinstance(node, dict):
            # Count filesort operations
            if node.get('using_filesort', False):
                features['filesort_count'] += 1
            
            # Count temporary tables
            if node.get('using_temporary_table', False):
                features['temp_table_count'] += 1
            
            # Process table information
            if 'table' in node:
                features['table_count'] += 1
                table_info = node['table']
                
                # Access type counts
                access_type = table_info.get('access_type', '').lower()
                if access_type == 'range':
                    features['range_access'] += 1
                elif access_type == 'ref':
                    features['ref_access'] += 1
                elif access_type == 'eq_ref':
                    features['eq_ref_access'] += 1
                elif access_type == 'all':
                    features['all_access'] += 1
                
                # Key length
                if 'key_length' in table_info:
                    features['total_key_length'] += safe_float(table_info['key_length'])
                
                # Filtered percentage
                if 'filtered' in table_info:
                    features['total_filtered_percent'] += safe_float(table_info['filtered'])
                
                # Accumulate costs
                if 'rows_examined_per_scan' in table_info:
                    features['total_rows_examined'] += safe_float(table_info['rows_examined_per_scan'])
                
                if 'cost_info' in table_info:
                    cost_info = table_info['cost_info']
                    features['total_read_cost'] += safe_float(cost_info.get('read_cost', 0))
                    features['total_eval_cost'] += safe_float(cost_info.get('eval_cost', 0))
                    features['total_prefix_cost'] += safe_float(cost_info.get('prefix_cost', 0))
            
            # Recursively process all values
            for value in node.values():
                traverse_plan(value)
        elif isinstance(node, list):
            for item in node:
                traverse_plan(item)
    
    traverse_plan(plan)
    
    # Normalize some features by table count if there are tables
    if features['table_count'] > 0:
        features['total_filtered_percent'] /= features['table_count']
    
    return list(features.values())

class QueryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class QueryNet(nn.Module):
    def __init__(self, input_size):
        super(QueryNet, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

def main():
    # Load data
    df = pd.read_csv('/workspace/data/query_costs.csv')
    print(f"Total number of queries: {len(df)}")
    
    # Extract features from JSON plans
    features = []
    for query_id in tqdm(df['query_id'], desc='Processing query plans'):
        with open(f'/workspace/data/row_plans/{query_id}.json', 'r') as f:
            plan = json.load(f)
            features.append(extract_features_from_plan(plan))
    
    X = np.array(features)
    y = df['use_imci'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Class distribution:\n{pd.Series(y).value_counts(normalize=True)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create data loaders
    train_dataset = QueryDataset(X_train, y_train)
    test_dataset = QueryDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = QueryNet(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model.to(device)
    
    best_accuracy = 0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            total_train += y_batch.size(0)
            correct_train += (predicted.squeeze() == y_batch).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch.unsqueeze(1)).item()
                predicted = (outputs.data > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted.squeeze() == y_batch).sum().item()
        
        accuracy = 100 * correct / total
        scheduler.step(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss/len(test_loader):.4f}, Val Accuracy: {accuracy:.2f}%')
        print(f'Best Accuracy: {best_accuracy:.2f}%')
        print('-' * 60)
        
        if patience_counter >= max_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break

if __name__ == '__main__':
    main()