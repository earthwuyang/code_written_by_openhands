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
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(',', '.'))
    except (ValueError, TypeError):
        return 0.0

def extract_features_from_plan(plan, plan_type='row'):
    if plan_type == 'row':
        return extract_row_plan_features(plan)
    else:
        return extract_column_plan_features(plan)

def extract_row_plan_features(plan):
    features = {
        'total_cost': 0.0,
        'table_count': 0,
        'total_rows': 0,
        'read_cost': 0,
        'eval_cost': 0,
        'prefix_cost': 0,
        'range_count': 0,
        'ref_count': 0,
        'eq_ref_count': 0,
        'full_scan_count': 0,
        'filesort_count': 0,
        'temp_table_count': 0,
        'key_length': 0,
        'filtered_percent': 0.0
    }
    
    if 'cost_info' in plan.get('query_block', {}):
        features['total_cost'] = safe_float(plan['query_block']['cost_info']['query_cost'])
    
    def traverse_row_plan(node):
        if isinstance(node, dict):
            if node.get('using_filesort', False):
                features['filesort_count'] += 1
            
            if node.get('using_temporary_table', False):
                features['temp_table_count'] += 1
            
            if 'table' in node:
                features['table_count'] += 1
                table_info = node['table']
                
                access_type = table_info.get('access_type', '').lower()
                if access_type == 'range':
                    features['range_count'] += 1
                elif access_type == 'ref':
                    features['ref_count'] += 1
                elif access_type == 'eq_ref':
                    features['eq_ref_count'] += 1
                elif access_type == 'all':
                    features['full_scan_count'] += 1
                
                features['key_length'] += safe_float(table_info.get('key_length', 0))
                features['filtered_percent'] += safe_float(table_info.get('filtered', 0))
                features['total_rows'] += safe_float(table_info.get('rows_examined_per_scan', 0))
                
                if 'cost_info' in table_info:
                    cost_info = table_info['cost_info']
                    features['read_cost'] += safe_float(cost_info.get('read_cost', 0))
                    features['eval_cost'] += safe_float(cost_info.get('eval_cost', 0))
                    features['prefix_cost'] += safe_float(cost_info.get('prefix_cost', 0))
            
            for value in node.values():
                traverse_row_plan(value)
        elif isinstance(node, list):
            for item in node:
                traverse_row_plan(item)
    
    traverse_row_plan(plan)
    
    if features['table_count'] > 0:
        features['filtered_percent'] /= features['table_count']
    
    return list(features.values())

def extract_column_plan_features(plan):
    features = {
        'total_cost': 0.0,
        'table_count': 0,
        'total_rows': 0,
        'read_cost': 0,
        'eval_cost': 0,
        'prefix_cost': 0,
        'range_count': 0,
        'ref_count': 0,
        'eq_ref_count': 0,
        'full_scan_count': 0,
        'filesort_count': 0,
        'temp_table_count': 0,
        'key_length': 0,
        'filtered_percent': 0.0
    }
    
    def traverse_column_plan(node):
        if isinstance(node, dict):
            node_str = str(node).lower()
            
            # Count operations
            if 'sort' in node_str:
                features['filesort_count'] += 1
            if 'temp_table' in node_str:
                features['temp_table_count'] += 1
            if 'table' in node_str:
                features['table_count'] += 1
            
            # Analyze access patterns
            if 'index' in node_str:
                features['range_count'] += 1
            elif 'join' in node_str:
                features['ref_count'] += 1
            elif 'scan' in node_str:
                features['full_scan_count'] += 1
            
            # Extract costs
            if 'Cost' in node:
                cost = safe_float(node['Cost'])
                features['total_cost'] += cost
                features['read_cost'] += cost * 0.7
                features['eval_cost'] += cost * 0.3
            
            # Extract rows if available
            if 'RowCount' in node:
                features['total_rows'] += safe_float(node['RowCount'])
            
            for value in node.values():
                traverse_column_plan(value)
        elif isinstance(node, list):
            for item in node:
                traverse_column_plan(item)
    
    traverse_column_plan(plan)
    return list(features.values())

class QueryDataset(Dataset):
    def __init__(self, X_row, X_col, y):
        self.X_row = torch.FloatTensor(X_row)
        self.X_col = torch.FloatTensor(X_col)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_row[idx], self.X_col[idx], self.y[idx]

class QueryNet(nn.Module):
    def __init__(self, input_size):
        super(QueryNet, self).__init__()
        
        # Feature extractors
        self.row_features = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU()
        )
        
        self.col_features = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU()
        )
        
        # Cost prediction heads
        self.row_cost = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
        
        self.col_cost = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x_row, x_col):
        # Extract features
        row_feats = self.row_features(x_row)
        col_feats = self.col_features(x_col)
        
        # Predict costs
        row_cost = self.row_cost(row_feats)
        col_cost = self.col_cost(col_feats)
        
        # Compare costs to make final decision
        return torch.sigmoid(col_cost - row_cost)

def main():
    print("Loading data...")
    data_dir = '/home/wuy/AI/MLR-Copilot/workspaces/row_column_routing/data/total_tpch_sf1'
    csv_file = os.path.join(data_dir, 'query_costs.csv')
    row_plan_dir = os.path.join(data_dir, 'row_plans')
    column_plan_dir = os.path.join(data_dir, 'column_plans')
    df = pd.read_csv(csv_file)
    print(f"Total number of queries: {len(df)}")
    
    print("\nExtracting features from plans...")
    row_features = []
    column_features = []
    
    for query_id in tqdm(df['query_id'], desc='Processing query plans'):
        # Extract row plan features
        with open(os.path.join(row_plan_dir, f'{query_id}.json'), 'r') as f:
            plan = json.load(f)
            row_features.append(extract_features_from_plan(plan, 'row'))
        
        # Extract column plan features
        with open(os.path.join(column_plan_dir, f'{query_id}.json'), 'r') as f:
            plan = json.load(f)
            column_features.append(extract_features_from_plan(plan, 'column'))
    
    X_row = np.array(row_features)
    X_col = np.array(column_features)
    y = df['use_imci'].values
    
    print("\nSplitting data...")
    # Split into train, validation, and test sets
    X_row_temp, X_row_test, X_col_temp, X_col_test, y_temp, y_test = train_test_split(
        X_row, X_col, y, test_size=0.2, random_state=42
    )
    
    X_row_train, X_row_val, X_col_train, X_col_val, y_train, y_val = train_test_split(
        X_row_temp, X_col_temp, y_temp, test_size=0.2, random_state=42
    )
    
    print("\nScaling features...")
    # Scale features
    row_scaler = StandardScaler()
    col_scaler = StandardScaler()
    
    X_row_train = row_scaler.fit_transform(X_row_train)
    X_row_val = row_scaler.transform(X_row_val)
    X_row_test = row_scaler.transform(X_row_test)
    
    X_col_train = col_scaler.fit_transform(X_col_train)
    X_col_val = col_scaler.transform(X_col_val)
    X_col_test = col_scaler.transform(X_col_test)
    
    # Create data loaders
    train_dataset = QueryDataset(X_row_train, X_col_train, y_train)
    val_dataset = QueryDataset(X_row_val, X_col_val, y_val)
    test_dataset = QueryDataset(X_row_test, X_col_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Get indices for test set to extract times later
    test_indices = np.arange(len(y))[len(X_row_temp):]
    test_row_times = df.iloc[test_indices]['row_time'].values
    test_column_times = df.iloc[test_indices]['column_time'].values
    
    print("\nInitializing model...")
    # Initialize model
    model = QueryNet(X_row_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    best_val_accuracy = 0
    best_val_f1 = 0
    patience_counter = 0
    max_patience = 10
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_preds = []
        train_labels = []
        
        # Training
        for X_row_batch, X_col_batch, y_batch in train_loader:
            X_row_batch = X_row_batch.to(device)
            X_col_batch = X_col_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_row_batch, X_col_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())
        
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_row_batch, X_col_batch, y_batch in val_loader:
                X_row_batch = X_row_batch.to(device)
                X_col_batch = X_col_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_row_batch, X_col_batch)
                val_loss += criterion(outputs, y_batch.unsqueeze(1)).item()
                predicted = (outputs.data > 0.5).float()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # Calculate metrics
        train_accuracy = 100 * np.mean(train_preds.squeeze() == train_labels)
        val_accuracy = 100 * np.mean(val_preds.squeeze() == val_labels)
        
        train_class_0_acc = 100 * np.mean((train_preds.squeeze() == 0) & (train_labels == 0))
        train_class_1_acc = 100 * np.mean((train_preds.squeeze() == 1) & (train_labels == 1))
        val_class_0_acc = 100 * np.mean((val_preds.squeeze() == 0) & (val_labels == 0))
        val_class_1_acc = 100 * np.mean((val_preds.squeeze() == 1) & (val_labels == 1))
        
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Train Class Accuracies - Row Store: {train_class_0_acc:.2f}%, Column Store: {train_class_1_acc:.2f}%')
        print(f'Val   - Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'Val Class Accuracies   - Row Store: {val_class_0_acc:.2f}%, Column Store: {val_class_1_acc:.2f}%')
        print(f'Best Val Accuracy: {best_val_accuracy:.2f}%')
        print('-' * 60)
        
        if patience_counter >= max_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Load best model for testing
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for X_row_batch, X_col_batch, y_batch in test_loader:
            X_row_batch = X_row_batch.to(device)
            X_col_batch = X_col_batch.to(device)
            outputs = model(X_row_batch, X_col_batch)
            predicted = (outputs.data > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(y_batch.numpy())
    
    test_preds = np.array(test_preds).squeeze()
    test_labels = np.array(test_labels)
    
    # Calculate metrics
    test_accuracy = 100 * np.mean(test_preds == test_labels)
    test_class_0_acc = 100 * np.mean((test_preds == 0) & (test_labels == 0))
    test_class_1_acc = 100 * np.mean((test_preds == 1) & (test_labels == 1))
    
    print('\nTest Results:')
    print(f'Overall Accuracy: {test_accuracy:.2f}%')
    print(f'Row Store Accuracy: {test_class_0_acc:.2f}%')
    print(f'Column Store Accuracy: {test_class_1_acc:.2f}%')
    
    # Calculate end-to-end runtime
    ai_total_time = sum([row_t if pred == 0 else col_t for pred, row_t, col_t in zip(test_preds, test_row_times, test_column_times)])
    optimal_total_time = sum([min(row_t, col_t) for row_t, col_t in zip(test_row_times, test_column_times)])
    actual_total_time = sum([row_t if label == 0 else col_t for label, row_t, col_t in zip(test_labels, test_row_times, test_column_times)])
    
    # Calculate cost threshold based predictions
    cost_threshold = 50000
    threshold_preds = []
    for query_id in df.iloc[test_indices]['query_id']:
        data_dir = '/home/wuy/AI/MLR-Copilot/workspaces/row_column_routing/data/total_tpch_sf1'
        row_plan_dir = os.path.join(data_dir, 'row_plans')
        with open(os.path.join(row_plan_dir, f'{query_id}.json'), 'r') as f:
            plan = json.load(f)
            if 'cost_info' in plan.get('query_block', {}):
                cost = safe_float(plan['query_block']['cost_info']['query_cost'])
                threshold_preds.append(1 if cost > cost_threshold else 0)
            else:
                threshold_preds.append(0)
    
    threshold_total_time = sum([row_t if pred == 0 else col_t for pred, row_t, col_t in zip(threshold_preds, test_row_times, test_column_times)])
    
    print('\nEnd-to-end Runtime Results:')
    print(f'AI Model Total Time: {ai_total_time:.4f} seconds')
    print(f'Cost Threshold Total Time: {threshold_total_time:.4f} seconds')
    print(f'Optimal Total Time: {optimal_total_time:.4f} seconds')
    print(f'Actual Total Time: {actual_total_time:.4f} seconds')
    
    print('\nRelative Performance:')
    print(f'AI Model vs Optimal: {(ai_total_time/optimal_total_time - 1)*100:.2f}% slower than optimal')
    print(f'Cost Threshold vs Optimal: {(threshold_total_time/optimal_total_time - 1)*100:.2f}% slower than optimal')
    print(f'Actual vs Optimal: {(actual_total_time/optimal_total_time - 1)*100:.2f}% slower than optimal')

if __name__ == '__main__':
    main()