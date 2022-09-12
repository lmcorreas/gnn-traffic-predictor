import torch.nn as nn
import dgl

import defines

from trafficconvlayer import TrafficConvLayer, NUM_APPLY_EDGES_EXTRA_FEATURES, NUM_APPLY_NODES_EXTRA_FEATURES

class TrafficPredictor(nn.Module):
    def __init__(self, g, wrapper, in_features, out_features, n_hidden, n_gnnlayers, n_hiddengnn, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.wrapper = wrapper
        self.base_g=g
        self.out_features = out_features
        
        self.training_gnn = True
        
        graphs = [g for i in range(self.batch_size)]
        self.g = dgl.batch(graphs)
                
        self.layers = wrapper.create_sequential()
        
        # Linear input layer
        self.layers.append(wrapper.create_linear_layer(in_features, n_hidden))
        self.layers.append(wrapper.create_linear_layer(n_hidden, n_hidden))
        self.layers.append(wrapper.create_linear_layer(n_hidden, n_hidden))
        #self.layers.append(wrapper.create_dropout_layer())
        self.layers.append(wrapper.create_linear_layer(n_hidden, n_hidden))
        self.layers.append(wrapper.create_linear_layer(n_hidden, n_hidden))
        #self.layers.append(wrapper.create_dropout_layer())
        self.layers.append(wrapper.create_linear_layer(n_hidden, n_hidden))
        self.layers.append(wrapper.create_linear_layer(n_hidden, n_hidden))
        #self.layers.append(wrapper.create_dropout_layer())
        self.layers.append(wrapper.create_linear_layer(n_hidden, n_hidden))
        self.layers.append(wrapper.create_linear_layer(n_hidden, g.num_edges() * out_features))
                
        # out_features * (origen y destino)
        # 3: características arco       
        self.linear_a1 = wrapper.create_linear_layer(out_features * 2 + NUM_APPLY_EDGES_EXTRA_FEATURES + in_features, n_hiddengnn)
        self.linear_a2 = wrapper.create_linear_layer(n_hiddengnn, n_hiddengnn)
        #self.linear_ad = wrapper.create_dropout_layer()
        self.linear_a3 = wrapper.create_linear_layer(n_hiddengnn, n_hiddengnn)
        self.linear_a4 = wrapper.create_linear_layer(n_hiddengnn, n_hiddengnn)
        self.linear_a9 = wrapper.create_linear_layer(n_hiddengnn, out_features)
        self.linear_edges = [self.linear_a1, 
                             self.linear_a2,
                             #self.linear_ad,
                             self.linear_a3,
                             self.linear_a4,
                             self.linear_a9,
                             ]
        
        
        self.linear_b1 = wrapper.create_linear_layer(out_features + NUM_APPLY_EDGES_EXTRA_FEATURES + in_features, n_hiddengnn)
        self.linear_b2 = wrapper.create_linear_layer(n_hiddengnn, n_hiddengnn)
        #self.linear_bd = wrapper.create_dropout_layer()
        self.linear_b3 = wrapper.create_linear_layer(n_hiddengnn, n_hiddengnn)
        self.linear_b4 = wrapper.create_linear_layer(n_hiddengnn, n_hiddengnn)
        self.linear_b9 = wrapper.create_linear_layer(n_hiddengnn, out_features)
        self.linear_nodes = [self.linear_b1, 
                             self.linear_b2,
                             #self.linear_bd,
                             self.linear_b3,
                             self.linear_b4,
                             self.linear_b9,
                             ]
        
        self.linear_r1 = wrapper.create_linear_layer(out_features + NUM_APPLY_NODES_EXTRA_FEATURES + in_features, n_hiddengnn)
        self.linear_r2 = wrapper.create_linear_layer(n_hiddengnn, n_hiddengnn)
        #self.linear_rd = wrapper.create_dropout_layer()
        self.linear_r3 = wrapper.create_linear_layer(n_hiddengnn, n_hiddengnn)
        self.linear_r4 = wrapper.create_linear_layer(n_hiddengnn, n_hiddengnn)
        self.linear_r9 = wrapper.create_linear_layer(n_hiddengnn, out_features)
        self.linear_reduction = [self.linear_r1,
                                 self.linear_r2,
                                 #self.linear_rd,
                                 self.linear_r3,
                                 self.linear_r4,
                                 self.linear_r9,
                                ]
                        
        self.gnnlayers = wrapper.create_sequential()
        
        for i in range(n_gnnlayers - 1):
            self.gnnlayers.append(TrafficConvLayer(wrapper, self.linear_edges, self.linear_nodes, self.linear_reduction, self.batch_size, out_features))
            
        # La última devuelve directamente el valor de los arcos, en lugar de propagar a los nodos
        self.gnnlayers.append(TrafficConvLayer(wrapper, self.linear_edges, self.linear_nodes, self.linear_reduction, self.batch_size, out_features, last=True))
        
        #for i in range(n_gnnlayers - 1):
        #    self.gnnlayers.append(TrafficConv(wrapper, out_features, out_features, self.batch_size))
        
        #self.gnnlayers.append(EdgeDataPredictor(wrapper, out_features, out_features, self.batch_size))
        
    def get_batch_graph(self, size_of_batch):
        if size_of_batch == self.batch_size:
            return self.g
        elif size_of_batch == 1:
            return self.base_g
        
        # Different number, build a graph batch
        graphs = [self.base_g for i in range(size_of_batch)]
        return dgl.batch(graphs)

        
    def forward(self, features, y = None, apply_gnn = True):
        h = features
        
        # Linear layers
        for layer in self.layers:
            h = layer(h)

        # Ajustes del grafo para adaptarse al tamaño del batch
        # Básicamente, crea tantos grafos disjuntos como instancias presente el minilote, permitiendo
        # su procesamiento vectorizado en mayor medida
        graph = self.get_batch_graph(features.shape[0])
            
        h = self.wrapper.reshape_tensor(h, (graph.num_edges(), -1))
            
        if self.training_gnn and apply_gnn:
            inc_features = features[:,None,:].expand(-1, self.base_g.num_edges(), -1)
            inc_node_features = features[:,None,:].expand(-1, self.base_g.num_nodes(), -1)
            
            if defines.ONLY_COST_ON_GNN:
                h_gnn = [h.clone()]
            else:
                h_gnn = [h.clone().detach()]
                
            inc_features = self.wrapper.reshape_tensor(inc_features, (graph.num_edges(), -1))
            inc_node_features = self.wrapper.reshape_tensor(inc_node_features, (graph.num_nodes(), -1))
            
            if y is not None and defines.APPLY_LABELS_BEFORE_GNN:
                y_resh = self.wrapper.reshape_tensor(y, (graph.num_edges(), -1))
                labeled = y_resh>=0
                h_gnn[0].requires_grad_(True)
                h_gnn[0][:, :defines.NUM_CHECKED_VALUES][labeled] = y_resh[labeled]
                            
            # GNN layers
            for id, gnnlayer in enumerate(self.gnnlayers):
                h_gnn = gnnlayer(graph, (h_gnn[0], inc_features, inc_node_features))
                #if (not self.training) or (id >= defines.NUM_GNN_TO_TRAIN - 1):
                if (id >= defines.NUM_GNN_TO_TRAIN - 1):
                    break
        else:
            h_gnn = (h.detach(), None, None)
                
        return h, h_gnn[0], h_gnn[1], h_gnn[2]
    
    def set_training_gnn(self, value):
        self.training_gnn = value
        
        for layer in self.linear_edges:
            layer.requires_grad_(value)
            
        for layer in self.linear_nodes:
            layer.requires_grad_(value)
            
        for layer in self.linear_reduction:
            layer.requires_grad_(value)
            
    def set_training_linear(self, value):
        self.training_linear = value
        
        for layer in self.layers:
            layer.requires_grad_(value)
    
    def print_parameters(self):
        for i in range(0, len(self.gnnlayers)):
            for j in range(0, len(self.gnnlayers[i].linear_edges)):
                print(self.gnnlayers[i].linear_edges[j].bias.data)
                print(self.gnnlayers[i].linear_edges[j].weight.data)
                
            for j in range(0, len(self.gnnlayers[i].linear_nodes)):
                print(self.gnnlayers[i].linear_nodes[j].bias.data)
                print(self.gnnlayers[i].linear_nodes[j].weight.data)