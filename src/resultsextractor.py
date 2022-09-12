from calendar import calendar
import os
import numpy as np
import torch

os.environ['DGLBACKEND'] = "pytorch"

import defines
import random

random.seed(defines.RANDOM_SEED)
np.random.seed(defines.RANDOM_SEED)
torch.manual_seed(defines.RANDOM_SEED)

from backendadapter import BackendAdapter

from customloss import TrafficLoss

import matplotlib.pyplot as plt

def extract_results(model, dataset, dglgraph, num_real_edges):
    wrapper = BackendAdapter()
    
    results_l = []
    results_g = []
    
    
    results_l_abs = []
    results_g_abs = []
    
    results_l_abs_item = []
    results_g_abs_item = []
    
    res_num_label = []
    
    defines.GNN_MULT = 1
    defines.LINEAR_MULT = 1
    defines.OUT_LIMITS_FACTOR = 0
    
    tfcId = dglgraph.edata['tfcId']
    idxToNull = np.mod(list(range(len(tfcId))), 10)==0
    
    with torch.no_grad():    
        #x_test = wrapper.create_tensor(dataset.x_test).split(1)
        #y_test = wrapper.create_tensor(dataset.y_test).split(1)
        
        x_test = wrapper.create_tensor(dataset.x_val).split(1)
        y_test = wrapper.create_tensor(dataset.y_val).split(1)
                
        for d in zip(x_test, y_test):
            x = d[0]
            y = d[1]
            
            size_of_batch = x.shape[0]
            
            out, out_gnn, *_ = model(x)
            
            out = out.view(size_of_batch, dglgraph.num_edges(), -1)
            out_gnn = out_gnn.view(size_of_batch, dglgraph.num_edges(), -1)
            
            out_y_l = out[:,0:num_real_edges, 0:defines.NUM_CHECKED_VALUES]
            out_y_g = out_gnn[:,0:num_real_edges, 0:defines.NUM_CHECKED_VALUES]
                        
            if defines.VOID_SOME_POINTS_TRAINING:
                idxToNullTensor = wrapper.create_tensor(idxToNull[None, :, None])
                labeled_edges = ((y >= 0) & (idxToNullTensor > 0))
            else:
                labeled_edges = (y>=0)

            l = out_y_l[labeled_edges]
            g = out_y_g[labeled_edges]
            labeled_y = y[labeled_edges]
                
            
            results_l.append(tensor_to_numpy(l-labeled_y))
            results_g.append(tensor_to_numpy(g-labeled_y))
            
            
            results_l_abs.append(np.abs(tensor_to_numpy(l-labeled_y)))
            results_g_abs.append(np.abs(tensor_to_numpy(g-labeled_y)))
            
            results_l_abs_item.append(np.mean(np.abs(tensor_to_numpy(l-labeled_y))))
            results_g_abs_item.append(np.mean(np.abs(tensor_to_numpy(g-labeled_y))))
            
            num_labeled_edges = labeled_edges[labeled_edges].numel()
            res_num_label.append(num_labeled_edges)
        
    results_l = np.concatenate(results_l)
    results_g = np.concatenate(results_g)
    
    
    results_l_abs = np.concatenate(results_l_abs)
    results_g_abs = np.concatenate(results_g_abs)
    
    results_l_abs_item = np.array(results_l_abs_item)
    results_g_abs_item = np.array(results_g_abs_item)
    
    res_num_label = np.array(res_num_label)
    
    print('Media error parte lineal (item):', results_l_abs.mean())
    print('Media error parte lineal (muestra):', results_l_abs_item.mean())
    
    print('Media error GNN (item):', results_g_abs.mean())
    print('Media error GNN (muestra):', results_g_abs_item.mean())
    
    bp = plt.boxplot([results_l,results_g], labels=['Salida intermedia', 'Salida GNN'], showfliers=False)
    plt.grid(True, axis='y')
    plt.xlabel('Red estimadora')
    plt.ylabel('Error estimación')
    plt.ylim(-32, 32)
    plt.savefig('./' + defines.OUTPUT_BASE_PATH + 'statistics/boxplot.png')
    plt.show()
    plt.close()
    
    l = plt.scatter(res_num_label, results_l_abs_item)
    plt.xlabel('Número mediciones muestra')
    plt.ylabel('Error absoluto estimación')
    plt.ylim(0, 25)
    plt.xlim(550, 1200)
    plt.savefig('./' + defines.OUTPUT_BASE_PATH + 'statistics/resultsL.png')
    plt.show()
    plt.close()
    l.remove()
    plt.close()
    
    plt.scatter(res_num_label, results_g_abs_item)
    plt.xlabel('Número mediciones muestra')
    plt.ylabel('Error absoluto estimación')
    plt.ylim(0, 25)
    plt.xlim(550, 1200)
    plt.savefig('./' + defines.OUTPUT_BASE_PATH + 'statistics/resultsG.png')
    plt.show()
    
def tensor_to_numpy(tensor):
    return tensor.cpu().numpy()