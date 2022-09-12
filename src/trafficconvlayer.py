import dgl.function as fn
import torch.nn as nn

# 1- speed_kph
# 2- lanes
# 3- length
# 4- x diff
# 5- y diff
# 6- in_degree origin
# 7- out_degree origin
# 8- in_degree dest
# 9- out_degree dest
NUM_APPLY_EDGES_EXTRA_FEATURES = 9
        
# 1- inputLanes
# 2- outputLanes
# 3- x
# 4- y
NUM_APPLY_NODES_EXTRA_FEATURES = 4

class TrafficConvLayer(nn.Module):
    
    def __init__(self, wrapper, linear_edges, linear_nodes, linear_reduction, batch_size, feats, last=False):
        super(TrafficConvLayer, self).__init__()
        self.linear_edges = linear_edges
        self.linear_nodes = linear_nodes
        self.linear_reduction = linear_reduction
                
        self.wrapper = wrapper
        self.batch_size = batch_size
        self.feats = feats

        self.last = last
    
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        
        # Pasamos al modelo como entrada:
        # 1- Características del nodo origen
        # 2- Características del nodo destino
        # 3- Características físicas del arco
        score = self.wrapper.cat([
            h_u,
            h_v,
            edges.data['speed_kph'][:,None], 
            edges.data['lanes'][:,None],
            edges.data['length'][:,None],
            (edges.src['lat'] - edges.dst['lat'])[:,None],
            (edges.src['lon'] - edges.dst['lon'])[:,None],
            edges.data['origin_in_degree'][:,None],
            edges.data['origin_out_degree'][:,None],
            edges.data['dest_in_degree'][:,None],
            edges.data['dest_out_degree'][:,None],
            edges.data['input']
        ], 1)
        
        for l in self.linear_edges:
            score = l(score)
                    
        # El retorno es una serie de valores para cada arco (parametrizable), 
        # que será asignada a la variable indicada dentro de 'edata'
        return {'h_e': score}
        
    def upd_all(self, edges):
        # Recuperamos el valor almacenado en forward para usarlo junto al resto de características
        h_e = edges.data['h_e']
        
        # Concatenamos toda la información a contemplar
        score = self.wrapper.cat([
            h_e,
            edges.data['speed_kph'][:,None], 
            edges.data['lanes'][:,None],
            edges.data['length'][:,None],
            (edges.src['lat'] - edges.dst['lat'])[:,None],
            (edges.src['lon'] - edges.dst['lon'])[:,None],
            edges.data['origin_in_degree'][:,None],
            edges.data['origin_out_degree'][:,None],
            edges.data['dest_in_degree'][:,None],
            edges.data['dest_out_degree'][:,None],
            edges.data['input']
        ], 1)
        
        # Procesamiento en serie de capas lineales
        for l in self.linear_nodes:
            score = l(score)
        
        # El retorno es una serie de valores para cada arco (parametrizable), 
        # que será asignada a la variable indicada dentro de 'edata'
        return {'v': score}
        
    def forward(self, graph, h):
        with graph.local_scope():
            # Guardamos por separado el estado temporal y la información de entrada de la primera fase expandida
            graph.edata['h_e'], graph.edata['input'], graph.ndata['input'] = h
            
            # Lanza la actualización de la información en los nodos. Usa función sum para la agregación.
            graph.update_all(self.upd_all, fn.sum('v', 'h'))
            
            s = graph.ndata['h']

            # Concatenación de todos los datos de entrada 
            score = self.wrapper.cat([
                graph.ndata['h'],
                graph.ndata['input'],
                graph.ndata['lat'][:,None],
                graph.ndata['lon'][:,None],
                graph.ndata['inputLanes'][:,None], 
                graph.ndata['outputLanes'][:,None]
            ], 1)
            
            # La información pasa por una serie de capas lineales previamente a propagar a los arcos
            for l in self.linear_reduction:
                score = l(score)
                
            graph.ndata['h'] = score
            
            # Lanza la actualización de la propagación a los arcos
            graph.apply_edges(self.apply_edges)
                        
            # Retorna tanto la salida final a nivel de arco, como las intermedias de nodos y arcos
            return graph.edata['h_e'], score, s
