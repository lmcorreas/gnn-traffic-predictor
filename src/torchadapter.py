from matplotlib.pyplot import step
import torch
import torch.nn as nn

from torchdataset import TorchDataset
from torch.utils.data import DataLoader

import defines

class TorchAdapter():
    def __init__(self, activate_GPU):
        print('Checking CUDA')
        print('Available: ', torch.cuda.is_available(), ' - ',
              torch.cuda.device_count(), ' device(s)')
        
        if activate_GPU and torch.cuda.is_available:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        self.data_type = torch.float32
        if defines.USE_DOUBLE:
            self.data_type = torch.double

    def create_tensor(self, arr):
        tensor = torch.tensor(arr, device=self.device)
        if defines.USE_DOUBLE:
            return tensor.double()
        return tensor.float()
    
    def create_sequential(self):
        return nn.ModuleList()
    
    def create_linear_layer(self, in_features, out_features):
        return nn.Linear(in_features, out_features, device=self.device, dtype=self.data_type)
    
    def create_dropout_layer(self, p=0.5):
        return nn.Dropout(p)
        
    def reshape_tensor(self, tensor, size):
        return torch.reshape(tensor, size)
    
    def ones(self, size):
        return torch.ones(size, device=self.device, dtype=self.data_type)
    
    def cat(self, arr, dim):
        return torch.cat(arr, dim=dim)
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
        
    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        print('Model parameters loaded')
        
    def evaluate_model(self, model, loss, x, y, batch_size, num_edges, num_real_edges, train_gnn, deep_stats):
        model.eval()
        l_sum, n = 0.0, 0
        l_sum_linear = 0.0
        l_sum_gnn = 0.0
        
        with torch.no_grad():
            __x = torch.split(x, batch_size)
            __y = torch.split(y, batch_size)
            
            for xt, yt in zip(__x, __y):
                size_of_batch = xt.shape[0]
                labeled_edges = (yt>=0)
                
                if defines.RANDOMIZE_NUM_GNN:
                    defines.NUM_GNN_TO_TRAIN = (n % 5) + 1
                
                out, out_gnn, *_ = model(xt)
                
                y_pred = out.view(size_of_batch, num_edges, -1)
                y_pred = y_pred[:,0:num_real_edges,0:defines.NUM_CHECKED_VALUES]
                
                
                y_pred_gnn = out_gnn.view(size_of_batch, num_edges, -1)
                y_pred_gnn = y_pred_gnn[:,0:num_real_edges, 0:defines.NUM_CHECKED_VALUES]
                
                                
                l_l = loss(y_pred, yt, labeled_edges, defines.LINEAR_MULT)
                l_g = loss(y_pred_gnn, yt, labeled_edges, defines.GNN_MULT if train_gnn else 0.0)
                
                l = l_l + l_g
                
                l_sum += l.item()
                #n += size_of_batch
                #n += labeled_edges.sum().item()
                n+=1
                
                l_sum_linear += l_l.item()
                l_sum_gnn += (l_g.item() if train_gnn and defines.GNN_MULT > 0 else 0)
                
            if deep_stats:
                return l_sum / n, l_sum_linear / n, l_sum_gnn / n
                
            return l_sum / n
        
    def generate_dataloader(self, x, y, batch_size, shuffle):
        dataset = TorchDataset(x, y, self)
        
        return DataLoader(dataset, batch_size, shuffle, drop_last=True)
    
    def generate_optimizer(self, parameters, lr):
        return torch.optim.RMSprop(parameters, lr)
        
    def generate_scheduler(self, optimizer, step_size, gamma):
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=step_size,
                                            gamma=gamma)