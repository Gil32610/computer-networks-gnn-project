import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os.path as osp


class UNSWNB15Dataset(Dataset):
    def __init__(self, 
                 root,
                 num_neighbors,
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None, 
                 train=True
                 ):
        self.train = train
        self.num_neighbors = num_neighbors        
        self.original_data = self._load_data()
        self.num_samples = self.original_data.shape[0]
        super().__init__(root, transform, pre_transform, pre_filter)
        
        
    @property
    def raw_file_name(self):
        if self.train:
            return '../data/train/unsw_training-set_multi-label-preprocessing.csv'
        else :
            return 'data/test/unsw_test-set_multi-label-preprocessing.csv'
    
    @property
    def processed_file_names(self):
        split = 'train' if self.train else 'test'
        return [f'data_cluster_{split}_{i}.pt' for i in range(self.num_samples)]
    
    def _load_data(self):
      return pd.read_csv(self.raw_file_name, skiprows=1).values
        
    
    def process(self):
        neighborhood = NearestNeighbors(n_neighbors=self.num_neighbors, algorithm='auto') 
        features, labels = self.original_data[:,-10:], self.original_data[:,:-10]
        neighborhood.fit(features)
        
        for i in range(self.num_samples):
            indices = neighborhood.kneighbors(
                features[i].reshape(1,-1),
                return_distance=False
            )
            cluster_indices = indices.flatten()
            x_cluster = torch.tensor(features[cluster_indices], dtype=torch.float)
            y_node = torch.tensor(labels[i].reshape(1,-1), dtype=torch.float)
            
            num_nodes = x_cluster.size(0)
            src, dst = [], []
            for s in range(num_nodes):
                for d in range(num_nodes):
                    if s!=d:
                        src.append(s)
                        dst.append(d)
            
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            data = Data(x=x_cluster, edge_index=edge_index, y=y_node)
            
            path = osp.join(self.processed_dir,self.processed_file_names[i])
            torch.save(data, path)
            
    def __len__(self):
        return self.len()
            
    def len(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.get(idx)
    
    def get(self,idx):
        filename = self.processed_file_names[idx]
        path = osp.join(self.processed_dir,filename)
        data = torch.load(path)
        return data
        
        
        
        
    

