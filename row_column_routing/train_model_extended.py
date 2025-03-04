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
        return float(str(value).replace(',', '.'))
    except (ValueError, TypeError):
        return 0.0

def extract_features_from_plan(plan, plan_type='row'):
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
    
    def extract_row_plan_features(plan):
        if 'message' in plan.get('query_block', {}) and plan['query_block']['message'] == 'Impossible WHERE':
            features['impossible_where'] = 1
            return
        
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
                        features['range_access'] += 1
                    elif access_type == 'ref':
                        features['ref_access'] += 1
                    elif access_type == 'eq_ref':
                        features['eq_ref_access'] += 1
                    elif access_type == 'all':
                        features['all_access'] += 1
                    
                    if 'key_length' in table_info:
                        features['total_key_length'] += safe_float(table_info['key_length'])
                    
                    if 'filtered' in table_info:
                        features['total_filtered_percent'] += safe_float(table_info['filtered'])
                    
                    if 'rows_examined_per_scan' in table_info:
                        features['total_rows_examined'] += safe_float(table_info['rows_examined_per_scan'])
                    
                    if 'cost_info' in table_info:
                        cost_info = table_info['cost_info']
                        features['total_read_cost'] += safe_float(cost_info.get('read_cost', 0))
                        features['total_eval_cost'] += safe_float(cost_info.get('eval_cost', 0))
                        features['total_prefix_cost'] += safe_float(cost_info.get('prefix_cost', 0))
                
                for value in node.values():
                    traverse_row_plan(value)
            elif isinstance(node, list):
                for item in node:
                    traverse_row_plan(item)
        
        traverse_row_plan(plan)
    
    def extract_column_plan_features(plan):
        def traverse_column_plan(node):
            if isinstance(node, dict):
                # Count temporary tables
                if any(key.startswith('temp_table') for key in str(node).lower().split()):
                    features['temp_table_count'] += 1
                
                # Count sorts
                if any(key.startswith('sort') for key in node.keys()):
                    features['filesort_count'] += 1
                
                # Count table accesses
                if 'TableName' in node:
                    features['table_count'] += 1
                    table_name = node['TableName'].lower()
                    
                    # Estimate access types based on operation names
                    if 'index' in str(node).lower():
                        features['range_access'] += 1
                    elif 'join' in str(node).lower():
                        features['ref_access'] += 1
                    elif 'scan' in str(node).lower():
                        features['all_access'] += 1
                
                # Extract costs if available
                if 'Cost' in node:
                    cost = safe_float(node['Cost'])
                    features['total_cost'] += cost
                    features['total_read_cost'] += cost * 0.7  # Approximate read cost
                    features['total_eval_cost'] += cost * 0.3  # Approximate eval cost
                
                # Recursively process all values
                for value in node.values():
                    traverse_column_plan(value)
            elif isinstance(node, list):
                for item in node:
                    traverse_column_plan(item)
        
        traverse_column_plan(plan)
    
    if plan_type == 'row':
        extract_row_plan_features(plan)
    else:
        extract_column_plan_features(plan)
    
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
        self.row_net = nn.Sequential(
            nn.BatchNorm1d(input_size // 2),
            nn.Linear(input_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.col_net = nn.Sequential(
            nn.BatchNorm1d(input_size // 2),
            nn.Linear(input_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.combined = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Split input into row and column features
        row_features = x[:, :x.shape[1]//2]
        col_features = x[:, x.shape[1]//2:]
        
        # Process each type of features
        row_out = self.row_net(row_features)
        col_out = self.col_net(col_features)
        
        # Combine features
        combined = torch.cat([row_out, col_out], dim=1)
        return self.combined(combined)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs.data > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def calculate_end_to_end_time(predictions, actual_labels, row_times, column_times):
    ai_total_time = sum([row_t if pred == 0 else col_t for pred, row_t, col_t in zip(predictions, row_times, column_times)])
    optimal_total_time = sum([min(row_t, col_t) for row_t, col_t in zip(row_times, column_times)])
    actual_total_time = sum([row_t if label == 0 else col_t for label, row_t, col_t in zip(actual_labels, row_times, column_times)])
    
    return ai_total_time, optimal_total_time, actual_total_time

def cost_threshold_router(query_id, threshold=50000):
    try:
        with open(f'/workspace/data/row_plans/{query_id}.json', 'r') as f:
            plan = json.load(f)
            if 'cost_info' in plan['query_block']:
                cost = safe_float(plan['query_block']['cost_info']['query_cost'])
                return 1 if cost > threshold else 0
    except:
        return 0
    return 0

def main():
    # Load data
    df = pd.read_csv('/workspace/data/query_costs.csv')
    print(f"Total number of queries: {len(df)}")
    
    # Extract features from both row and column plans
    row_features = []
    column_features = []
    
    for query_id in tqdm(df['query_id'], desc='Processing query plans'):
        # Row plans
        with open(f'/workspace/data/row_plans/{query_id}.json', 'r') as f:
            plan = json.load(f)
            row_features.append(extract_features_from_plan(plan, 'row'))
        
        # Column plans
        with open(f'/workspace/data/column_plans/{query_id}.json', 'r') as f:
            plan = json.load(f)
            column_features.append(extract_features_from_plan(plan, 'column'))
    
    # Combine features
    X_row = np.array(row_features)
    X_col = np.array(column_features)
    X_combined = np.hstack([X_row, X_col])
    y = df['use_imci'].values
    
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    # Get corresponding times for test set
    test_indices = df.index[len(X_temp):]
    test_row_times = df.iloc[test_indices]['row_time'].values
    test_column_times = df.iloc[test_indices]['column_time'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create data loaders
    train_dataset = QueryDataset(X_train, y_train)
    val_dataset = QueryDataset(X_val, y_val)
    test_dataset = QueryDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Calculate class weights
    class_counts = np.bincount(y_train.astype(int))
    total_samples = len(y_train)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
    
    # Initialize model
    model = QueryNet(X_train.shape[1])
    criterion = nn.BCELoss(reduction='none')  # Use 'none' to apply weights per sample
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model.to(device)
    
    best_val_accuracy = 0
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
            # Apply class weights to the loss
            sample_weights = class_weights[y_batch.long()].to(device)
            losses = criterion(outputs, y_batch.unsqueeze(1))
            weighted_loss = (losses * sample_weights.unsqueeze(1)).mean()
            weighted_loss.backward()
            optimizer.step()
            
            total_loss += weighted_loss.item()
            predicted = (outputs.data > 0.5).float()
            total_train += y_batch.size(0)
            correct_train += (predicted.squeeze() == y_batch).sum().item()
            
            # Print batch statistics
            if total_train % 1000 == 0:
                print(f'  Batch stats - Loss: {weighted_loss.item():.4f}, '
                      f'Accuracy: {100 * correct_train/total_train:.2f}%, '
                      f'Pred dist: {predicted.mean().item():.2f}')
        
        train_accuracy = 100 * correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                sample_weights = class_weights[y_batch.long()].to(device)
                losses = criterion(outputs, y_batch.unsqueeze(1))
                weighted_loss = (losses * sample_weights.unsqueeze(1)).mean()
                val_loss += weighted_loss.item()
                
                predicted = (outputs.data > 0.5).float()
                total_val += y_batch.size(0)
                correct_val += (predicted.squeeze() == y_batch).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        val_accuracy = 100 * correct_val / total_val
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # Calculate class-wise accuracy
        class_0_acc = 100 * np.mean((val_preds == 0) & (val_labels == 0))
        class_1_acc = 100 * np.mean((val_preds == 1) & (val_labels == 1))
        
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Val   - Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'Class Accuracies - Row Store: {class_0_acc:.2f}%, Column Store: {class_1_acc:.2f}%')
        print(f'Best Val Accuracy: {best_val_accuracy:.2f}%')
        print('-' * 60)
        
        if patience_counter >= max_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on test set
    test_preds, test_labels = evaluate_model(model, test_loader, device)
    test_accuracy = 100 * np.mean(test_preds == test_labels)
    print(f'\nTest Accuracy: {test_accuracy:.2f}%')
    
    # Calculate end-to-end runtime for AI model
    ai_time, optimal_time, actual_time = calculate_end_to_end_time(
        test_preds, test_labels, test_row_times, test_column_times
    )
    
    # Calculate cost threshold based predictions
    cost_threshold_preds = [cost_threshold_router(qid) for qid in df.iloc[test_indices]['query_id']]
    threshold_time, _, _ = calculate_end_to_end_time(
        cost_threshold_preds, test_labels, test_row_times, test_column_times
    )
    
    print('\nEnd-to-end Runtime Results:')
    print(f'AI Model Total Time: {ai_time:.4f} seconds')
    print(f'Cost Threshold Total Time: {threshold_time:.4f} seconds')
    print(f'Optimal Total Time: {optimal_time:.4f} seconds')
    print(f'Actual Total Time: {actual_time:.4f} seconds')
    
    print('\nRelative Performance:')
    print(f'AI Model vs Optimal: {(ai_time/optimal_time - 1)*100:.2f}% slower than optimal')
    print(f'Cost Threshold vs Optimal: {(threshold_time/optimal_time - 1)*100:.2f}% slower than optimal')
    print(f'Actual vs Optimal: {(actual_time/optimal_time - 1)*100:.2f}% slower than optimal')

if __name__ == '__main__':
    main()