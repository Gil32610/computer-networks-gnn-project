from network.dataset import UNSWNB15Dataset
from models.sage_ids import SAGEAttackClassifier
from torch_geometric.data import Batch
import torch

if __name__ == '__main__':
    root = "../data/test"
    in_channels = 42
    hidden_channels = 10
    num_classes = 10
    
    dataset = UNSWNB15Dataset(
        root=root,
        num_neighbors=5,
        train=False,
    )
    input = dataset[0]
    
    print(f"Loading single instance: {input}")
    print(f"Features shape: {input.x.shape}")
    print(f"Labels shape: {input.y.shape}")
    
    model = SAGEAttackClassifier(in_channels=in_channels, hidden_channels=hidden_channels, num_classes=num_classes)
    model.eval()
    batch_data = Batch.from_data_list([input])
    
    x = batch_data.x
    edge_index = batch_data.edge_index
    batch = batch_data.batch
    
    with torch.no_grad():
        output = model(x, edge_index, batch)
        
    print(f"Output shape: {output.shape}")
    print(f"Probabilities for 10 labels: \n{output.squeeze(0)}" )
