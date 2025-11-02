from torch_geometric.loader import DataLoader
from network.dataset import UNSWNB15Dataset
from models.sage_ids import SAGEAttackClassifier
from train import run_training
import torch.optim as optim
import torch 
import os

BATCH_SIZE = 32
NUM_NEIGHBORS = 5
IN_CHANNELS = 42
HIDDEN_CHANNELS = 32
NUM_CLASSES = 10


if __name__ == '__main__':
    
    train_dataset = UNSWNB15Dataset(
        root='../data/train',
        num_neighbors=NUM_NEIGHBORS,
        train=True
        )
    
    test_dataset = UNSWNB15Dataset(
        root='../data/test',
        num_neighbors=NUM_NEIGHBORS,
        train=False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2 or 1
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1
    )
    
    model = SAGEAttackClassifier(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=NUM_CLASSES
        )
    
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=1e-3,
        weight_decay=5e-4
        )
    
    save_dir = '../data/outputs'
    
    run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=100,
        save_dir=save_dir)