from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CRITERION = nn.BCELoss()



def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_proba = y_pred.cpu().numpy()
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    y_true_np = y_true.cpu().numpy()
    
    return {
        'micro_f1': f1_score(y_true_np, y_pred_binary, average='micro', zero_division=0),
        'micro_recall': recall_score(y_true_np, y_pred_binary, average='micro', zero_division=0),
        'micro_precision': precision_score(y_true_np, y_pred_binary, average='micro', zero_division=0),
        'exact_accuracy': accuracy_score(y_true_np, y_pred_binary),
    }
    
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_targets = []
    all_probabilities = []
    
    
    for batch_data in loader:
        batch_data = batch_data.to(DEVICE)
        optimizer.zero_grad()
        
        
        probabilities = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        
        
        loss = criterion(probabilities, batch_data.y.float())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_data.num_graphs
        
        all_targets.append(batch_data.y)
        all_probabilities.append(probabilities)
        
    avg_loss = total_loss / len(loader.dataset)
    targets = torch.cat(all_targets, dim=0)
    probabilities = torch.cat(all_probabilities, dim=0)
    metrics = calculate_metrics(targets, probabilities)
    
    metrics['loss'] = avg_loss
    return metrics

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_targets = []
    all_probabilities = []
    
    for batch_data in loader:
        batch_data = batch_data.to(DEVICE)
        probabilities = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        
        loss = criterion(probabilities, batch_data.y.float())
        total_loss += loss.item() * batch_data.num_graphs
        
        all_targets.append(batch_data.y)
        all_probabilities.append(probabilities)
    
    avg_loss = total_loss / len(loader.dataset)
    
    targets = torch.cat(all_targets, dim=0)
    probabilities = torch.cat(all_probabilities, dim=0)
    
    metrics = calculate_metrics(targets, probabilities)
    metrics['loss'] = avg_loss
    
    return metrics

def run_training(model, train_loader, test_loader, optimizer, epochs, save_dir, metric_key='micro_f1', criterion=CRITERION):
    history = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'test_micro_f1': [],
        'test_micro_recall': [],
        'test_micro_precision': [],
        'test_exact_accuracy': [],
    }
    best_metric_value = -float('inf')
    best_model_state = None
    
    model.to(DEVICE)
    print(f"Starting training on device: {DEVICE}")

    os.makedirs(save_dir, exist_ok=True)
    
    model_save_path = os.path.join(save_dir, 'best_gnn_model.pth')
    history_save_path = os.path.join(save_dir, 'training_history.csv')

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion)
        test_metrics = evaluate(model, test_loader, criterion)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_micro_f1'].append(test_metrics['micro_f1'])
        history['test_micro_recall'].append(test_metrics['micro_recall'])
        history['test_micro_precision'].append(test_metrics['micro_precision'])
        history['test_exact_accuracy'].append(test_metrics['exact_accuracy'])
        
        current_metric_value = test_metrics[metric_key]
        
        
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_model_state = deepcopy(model.state_dict())
            
            print(f"Epoch {epoch:03d}: NEW BEST! Test {metric_key.upper()}: {best_metric_value:.4f}")

        print(f"Epoch {epoch:03d} | Train Loss: {train_metrics['loss']:.4f} | Test Loss: {test_metrics['loss']:.4f} | Test F1: {test_metrics['micro_f1']:.4f}")


    if best_model_state:
        torch.save({
            'model_state_dict': best_model_state,
            'best_metric_value': best_metric_value,
            'metric_key': metric_key,
        }, model_save_path)
        print(f"\nBest model checkpoint saved to: {model_save_path}")


    history_df = pd.DataFrame(history)
    history_df.to_csv(history_save_path, index=False)
    print(f"Training history saved to: {history_save_path}")


    return history_df
    
    


