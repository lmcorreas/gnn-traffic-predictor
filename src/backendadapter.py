import defines

import torchadapter as torchAdap

class BackendAdapter():
    def __init__(self):
        if defines.BACKEND == 'PYTORCH':
            self.backend = torchAdap.TorchAdapter(defines.ACTIVATE_GPU)
            
    def get_device(self):
        return self.backend.device
            
    def create_tensor(self, arr):
        return self.backend.create_tensor(arr)
    
    def create_sequential(self):
        return self.backend.create_sequential()
    
    def create_linear_layer(self, in_features, out_features):
        return self.backend.create_linear_layer(in_features, out_features)
        
    def create_dropout_layer(self, p=0.5):
        return self.backend.create_dropout_layer(p)
    
    def reshape_tensor(self, tensor, size):
        return self.backend.reshape_tensor(tensor, size)
    
    def ones(self, size):
        return self.backend.ones(size)
    
    def cat(self, arr, dim):
        return self.backend.cat(arr, dim)
    
    def save_model(self, model):
        path = defines.MODEL_VOIDING_PARAMS_PATH if defines.VOID_SOME_POINTS_TRAINING else defines.MODEL_PARAMS_PATH
        if defines.SECOND_MAP:
            path = defines.MODEL_SAN_JOSE        
        
        self.backend.save_model(model, path)
        
    def load_model(self, model):
        path = defines.MODEL_VOIDING_PARAMS_PATH if defines.VOID_SOME_POINTS_TRAINING else defines.MODEL_PARAMS_PATH
        if defines.SECOND_MAP:
            path = defines.MODEL_SAN_JOSE
        self.backend.load_model(model, path)
        
    def evaluate_model(self, model, loss, x, y, batch_size, num_edges, num_real_edges, train_gnn, deep_stats=False):
        return self.backend.evaluate_model(model, loss, x, y, batch_size, num_edges, num_real_edges, train_gnn, deep_stats)
    
    def generate_dataloader(self, x, y, batch_size, shuffle):
        return self.backend.generate_dataloader(x, y, batch_size, shuffle)
    
    def generate_optimizer(self, parameters, lr):
        return self.backend.generate_optimizer(parameters, lr)
    
    def generate_scheduler(self, optimizer, step_size, gamma):
        return self.backend.generate_scheduler(optimizer, step_size, gamma)