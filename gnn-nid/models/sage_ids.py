import torch 
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch.nn import BatchNorm1d, ReLU

class SAGEAttackClassifier(torch.nn.Module):
    
    def __init__(self,in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=in_channels, out_channels=hidden_channels)
        self.batch_norm1 = BatchNorm1d(hidden_channels)
        self.act1 = ReLU()
        self.conv2 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.batch_norm2 = BatchNorm1d(hidden_channels)
        self.act2 = ReLU()
        self.linear = torch.nn.Linear(in_features=hidden_channels,out_features=num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = self.act2(x)
        x = global_mean_pool(x, batch=batch)
        x = self.linear(x)
        x = torch.sigmoid(x)
        
        