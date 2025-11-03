import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from typing import List, Optional


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_metrics_and_save(
    csv_filepath: str,
    output_filepath: str = 'metrics_plot.png',  
    metrics_to_plot: Optional[List[str]] = None,
    epoch_column: str = 'epoch',
    title: str = "Model Training and Evaluation Metrics Over Epochs",
    use_seaborn: bool = True
):
    
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at path: {csv_filepath}")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the CSV: {e}")
        return

    if use_seaborn:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use('ggplot')

    # --- Select Columns to Plot ---
    if metrics_to_plot is None:
        plot_columns = [col for col in df.columns if col != epoch_column]
    else:
        plot_columns = [m for m in metrics_to_plot if m in df.columns]
        if len(plot_columns) < len(metrics_to_plot):
            missing = set(metrics_to_plot) - set(plot_columns)
            print(f"⚠️ Warning: Metrics missing from CSV and skipped: {missing}")

    if not plot_columns:
        print("❌ Error: No valid metrics columns to plot.")
        return

    
    plt.figure(figsize=(12, 6))

    for metric in plot_columns:
        plt.plot(df[epoch_column], df[metric], marker='o', linestyle='-', label=metric)

    
    plt.title(title, fontsize=16)
    plt.xlabel(epoch_column.capitalize(), fontsize=14)
    plt.ylabel("Metric Value / Loss", fontsize=14)
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    try:
        plt.savefig(output_filepath, bbox_inches='tight')
        print(f"✅ Plot successfully saved to: **{output_filepath}**")
    except Exception as e:
        print(f"❌ Error saving plot to {output_filepath}: {e}")

    plt.close()
    
def plot_roc_curve(
    save_path,
    model: torch.nn.Module,
    test_loader: DataLoader,
    device = DEVICE,
    title='ROC Curve'
):
    model.eval()
    all_preds=[]
    all_labels=[]
    model.to(device=device)
    with torch.no_grad:
        for inputs, labels in test_loader:
            inputs = inputs.to(device=device)
            outputs = model(inputs)
            
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
            elif outputs.dim() > 1 and outputs.shape[1] == 1:
                probabilities = torch.sigmoid(outputs).squeeze()
            else:
                probabilities = outputs.squeeze()
                
            all_preds.extend(probabilities.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
    true_labels = np.array(all_labels)
    predicted_scores = np.array(all_preds)
    
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(fpr,tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr, tpr, color='darkorange', lw=2,
        label=f'ROC curve (area = {roc_auc:.4f})'
    )
    
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)    
    plt.savefig(f'{save_path}/{title}.png')