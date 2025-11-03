from plot import plot_metrics_and_save, plot_roc_curve
from models import SAGEAttackClassifier
from network.dataset import UNSWNB15Dataset
from torch_geometric.loader import DataLoader
import torch
import os

BATCH_SIZE = 32
NUM_NEIGHBORS = 5
IN_CHANNELS = 42
HIDDEN_CHANNELS = 32
NUM_CLASSES = 10


if __name__ == '__main__':
    csv_file_path = '../data/outputs/training_history.csv'
    output_path = '../data/plots/sage_plot'
    title = 'Graph Sage Network Intrusion Classification Metrics'
    plot_metrics_and_save(csv_filepath=csv_file_path,output_filepath=output_path,title=title, metrics_to_plot=['test_micro_f1','test_loss','train_loss'])
    model = SAGEAttackClassifier(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, num_classes=NUM_CLASSES)
    trained_model_path = '../data/outputs/best_gnn_model.pth'
    state_dict = torch.load(trained_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict=state_dict['model_state_dict'])
    test_dataset = UNSWNB15Dataset(root='../data/test/', num_neighbors=5, train=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1
        )
    plot_roc_curve(save_path=output_path, model=model,test_loader=test_loader,title='SAGE Attack Classifier')
    
    