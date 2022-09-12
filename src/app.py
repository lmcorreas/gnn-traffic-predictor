from datetime import datetime
import os
import numpy as np
import torch

os.environ['DGLBACKEND'] = "pytorch"

import defines
import random

random.seed(defines.RANDOM_SEED)
np.random.seed(defines.RANDOM_SEED)
torch.manual_seed(defines.RANDOM_SEED)

import dgl

from map import Map
from backendadapter import BackendAdapter
from trafficmodel import TrafficPredictor
from dataset import TrafficDataset

from customloss import TrafficLoss

from calendarreader import CalendarReader

from mapplotting import print_map

from resultsextractor import extract_results

class App():
    def init_app(self):        
        self.wrapper = BackendAdapter()
        
        self.load_map()
        loaded = self.init_model()
                        
        if loaded:
            self.load_simplified_data()
            self.load_data()
            extract_results(self.model, self.dataset, self.dglgraph, self.num_real_edges)
            self.draw_test_data()
        else:
            self.load_data()
            self.train_model()
        
    def load_map(self):
        self.map = Map()
        self.graph, self.fullgraph = self.map.load_map()
        
        self.relevantPointIds = self.map.filter_traffic_control_points()
        print('Hallados ', len(self.relevantPointIds), ' puntos de medida relevantes')
        
        # Ajustamos valores propiedades a entero, para evitar fallos al generar el grafo DGL
        self.map.preprocess_graph()
        
        self.dglgraph = dgl.from_networkx(self.graph, ['lat', 'lon', 'inputLanes', 'outputLanes'],
                                                    ['tfcId', 'speed_kph', 'lanes', 'length',
                                                        'origin_in_degree', 'origin_out_degree',
                                                        'dest_in_degree', 'dest_out_degree'],
                                          device=self.wrapper.get_device())
        self.num_real_edges = self.dglgraph.num_edges()
    
    def init_model(self):
        self.model = TrafficPredictor(self.dglgraph, self.wrapper, 
                       in_features=defines.NUM_INPUT_FEATURES, 
                       out_features=defines.NUM_OUTPUT_VALUES,
                       n_hidden=defines.NUM_HIDDEN_FEATURES_LINEAR,
                       n_gnnlayers=defines.NUM_GNN_LAYERS,
                       n_hiddengnn=defines.NUM_HIDDEN_FEATURES_GNN,
                       batch_size=defines.BATCH_SIZE)
        
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total parámetros modelo: ', params)
        
        if defines.LOAD_MODEL:
            try:
                self.wrapper.load_model(self.model)
                
                return True
            except:
                print('ERROR LOADING MODEL PARAMETERS')                
        
        return False
    
    def get_point_traffic_info(self, latStr, lonStr, yearStr, monthStr, dayStr, hourStr, minuteStr):
        if self.model is None:
            return None
        else:
            lat = float(latStr)
            lon = float(lonStr)
            
            input, using_now = self.prepare_api_input_data(yearStr, monthStr, dayStr, hourStr, minuteStr)
            
            inputTensor = self.wrapper.create_tensor(input[None, :])
            
            output = self.model(inputTensor)
            
            # output contiene la salida completa, ambas fases
            # Hacer traducción de lat,lon a arco concreto y devolver info del mismo
            edge_vert = self.map.get_edge_from_latlon(lat, lon)
            
            result = {
                'request': {
                    'lat': latStr,
                    'lon': lonStr,
                    'year': yearStr,
                    'month': monthStr,
                    'day': dayStr,
                    'hour': hourStr,
                    'minute': minuteStr,
                    'using_now': using_now
                },
                'value': None
            }
            
            if edge_vert[0] >= 0 and edge_vert[1] >= 0:
                edge_info = self.graph[edge_vert[0]][edge_vert[1]][0]
            
                estimation = round(min(max(output[1][edge_info['idx']].item(), 0), 100), 2)
                                
                result['value'] = {
                    'idx': edge_info['idx'],
                    'name': edge_info['name'],
                    'estimation': estimation
                }
            
            return result
        
    def get_traffic_info(self, yearStr, monthStr, dayStr, hourStr, minuteStr):
        if self.model is None:
            return None
        else:
            input, using_now = self.prepare_api_input_data(yearStr, monthStr, dayStr, hourStr, minuteStr)
            
            inputTensor = self.wrapper.create_tensor(input[None, :])
            
            output = self.model(inputTensor)
            
            result = {
                'request': {
                    'year': yearStr,
                    'month': monthStr,
                    'day': dayStr,
                    'hour': hourStr,
                    'minute': minuteStr,
                    'using_now': using_now
                },
                'value': []
            }
            
            for s,d,_ in self.graph.edges.data():
                edge_info = self.graph[s][d][0]
                
                estimation = round(min(max(output[1][edge_info['idx']].item(), 0), 100), 2)
                                
                value = {
                    'idx': edge_info['idx'],
                    'name': edge_info['name'] if 'name' in edge_info else '',
                    'estimation': estimation
                }
                
                result['value'].append(value)
                
            return result
    
    def prepare_api_input_data(self, yearStr, monthStr, dayStr, hourStr, minuteStr):
        # Si no están especificados correctamente los parámetros temporales, usamos 'now'
        dt = datetime.now
        using_now = True
        if None not in (yearStr, monthStr, dayStr, hourStr, minuteStr):
            year = int(yearStr)
            month = int(monthStr)
            day = int(dayStr)
            hour = int(hourStr)
            minute = int(minuteStr)
            
            try:
                dt = datetime(year, month, day, hour, minute)
                using_now = False
            except:
                print('ERROR creating datetime, using now as input')
                        
        dayofweek = dt.weekday()
        
        dateStr = dt.strftime('%d/%m/%Y')                
        holiday = self.calendarReader.get_data_value(dateStr)
        
        input = np.zeros(defines.NUM_INPUT_FEATURES)
        
        input[dayofweek] = 1
        input[6] = dt.day / defines.MAX_DAY_VALUE
        input[7] = dt.month / defines.MAX_MONTH_VALUE
        input[8] = dt.year / defines.MAX_YEAR_VALUE
        input[9] = dt.hour / defines.MAX_HOUR_VALUE
        input[10] = dt.minute / defines.MAX_MINUTE_VALUE
        input[11] = holiday[0]
        input[12] = ((input[9] * 60) + input[10]) / defines.MAX_DAY_MINUTES_VALUE
        input[13] = (1440 - input[12]) / defines.MAX_DAY_MINUTES_VALUE
        
        d = np.datetime64(dt)
        
        prevDt = d - np.timedelta64(1, 'D')
        calPrevData = self.calendarReader.get_data_value(prevDt)
        input[14] = calPrevData[0]
        
        nextDt = d + np.timedelta64(1, 'D')
        calNextData = self.calendarReader.get_data_value(nextDt)
        input[15] = calNextData[0]
        
        monthIdx = int(15+input[7])
        input[monthIdx] = 1
        
        return input, using_now
    
    def load_data(self):
        self.dataset = TrafficDataset(self.dglgraph.edata['tfcId'], self.relevantPointIds)
        
        # Anulamos mediciones de algunos puntos al azar, para comprobar GNN
        if defines.VOID_SOME_POINTS_TRAINING:
            indicesToVoid = self.PointIndicesToNull()
            self.dataset.y_train_copy[:, indicesToVoid, 0] = -1
            self.train_data_loader = self.wrapper.generate_dataloader(self.dataset.x_train_copy, self.dataset.y_train_copy, defines.BATCH_SIZE, defines.SHUFFLE_TRAIN_DATA)
            self.reduced_train_data_loader = self.wrapper.generate_dataloader(self.dataset.x_train_copy, self.dataset.y_train_copy, 8, defines.SHUFFLE_TRAIN_DATA)
        else:
            self.train_data_loader = self.wrapper.generate_dataloader(self.dataset.x_train, self.dataset.y_train, defines.BATCH_SIZE, defines.SHUFFLE_TRAIN_DATA)
            self.reduced_train_data_loader = self.wrapper.generate_dataloader(self.dataset.x_train, self.dataset.y_train, 8, defines.SHUFFLE_TRAIN_DATA)
        
        self.x_val = self.wrapper.create_tensor(self.dataset.x_val)
        self.y_val = self.wrapper.create_tensor(self.dataset.y_val)
        
    def load_simplified_data(self):
        self.calendarReader = CalendarReader()
        
    def train_model(self):
        self.optimizer = self.wrapper.generate_optimizer(self.model.parameters(), lr=defines.LEARNING_RATE)
        self.optimizer_gnn = self.wrapper.generate_optimizer(self.model.parameters(), lr=defines.LEARNING_RATE_GNN)
        self.loss = TrafficLoss(out_limits_factor=defines.OUT_LIMITS_FACTOR)
        
        self.scheduler = self.wrapper.generate_scheduler(self.optimizer,
                                            step_size=defines.DECAY_STEP,
                                            gamma=defines.GAMMA)
        
        min_val_loss = np.inf
        
        last_epoch_loss = -1
        last_val_epoch = 0
        
        # Si hay 0 sin GNN, activamos desde el principio
        train_gnn = defines.EPOCHS_WITHOUT_GNN <= 0
        
        if defines.RECALC_MODEL:
            self.wrapper.load_model(self.model)
            if defines.EPOCHS_WITHOUT_GNN < 1000:
                train_gnn = True
                self.model.set_training_gnn(train_gnn)
                self.optimizer = self.wrapper.generate_optimizer(self.model.parameters(), lr=defines.LEARNING_RATE)
                self.scheduler = self.wrapper.generate_scheduler(self.optimizer,
                                                step_size=defines.DECAY_STEP,
                                                gamma=defines.GAMMA)
                self.optimizer_gnn = self.wrapper.generate_optimizer(self.model.parameters(), lr=defines.LEARNING_RATE_GNN)
            val_loss_load = self.wrapper.evaluate_model(self.model, self.loss, self.x_val, self.y_val, defines.BATCH_SIZE, self.dglgraph.num_edges(), self.num_real_edges, train_gnn, deep_stats=True)
            min_val_loss = val_loss_load[0]
            print("loaded model", "validation loss:", val_loss_load[0], "(", val_loss_load[1], ",", val_loss_load[2], ")")
            
        int_loss = torch.nn.MSELoss()
                
        for epoch in range(1, defines.NUM_EPOCHS + 1):
            l_sum, l_sum1, l_sum2, n = 0.0, 0.0, 0.0, 0
            l_mid = 0.0
            self.model.train()
            
            self.model.set_training_gnn(train_gnn)
            #self.model.set_training_linear(not train_gnn)
            self.model.set_training_linear(True)
            
            seen_instances = 0
            
            
            for xt, yt in self.train_data_loader: #if not train_gnn else self.reduced_train_data_loader:
                size_of_batch = xt.shape[0]
                
                labeled_edges = (yt>=0)
                
                if defines.RANDOMIZE_NUM_GNN:
                    defines.NUM_GNN_TO_TRAIN = random.randint(1, 5)
                
                out, out_gnn, nodes_pred, nodes_sum = self.model(xt, yt, apply_gnn = False)
                
                y_pred = out.view(size_of_batch, self.dglgraph.num_edges(), -1)
                y_pred = y_pred[:,0:self.num_real_edges,0:defines.NUM_CHECKED_VALUES]
                
                l_l = self.loss(y_pred, yt, labeled_edges, defines.LINEAR_MULT)
                
                self.optimizer.zero_grad()
                    
                l_l.backward()
                self.optimizer.step()
                
                
                out, out_gnn, nodes_pred, nodes_sum = self.model(xt, yt, apply_gnn = True)
                                
                y_pred_gnn = out_gnn.view(size_of_batch, self.dglgraph.num_edges(), -1)
                y_pred_gnn = y_pred_gnn[:,0:self.num_real_edges,0:defines.NUM_CHECKED_VALUES]
                                
                l_g = self.loss(y_pred_gnn, yt, labeled_edges, defines.GNN_MULT if train_gnn else 0)
                
                if train_gnn and defines.INTER_MULT > 0:
                    nodes_pred = nodes_pred[:, 0:defines.NUM_CHECKED_VALUES]
                    a = torch.clamp(nodes_pred, 0, 100)
                    nodes_pred_l = int_loss(nodes_pred, a) * defines.INTER_MULT
                    l += nodes_pred_l
                    
                    nodes_sum = nodes_sum[:, 0:defines.NUM_CHECKED_VALUES]
                    b = torch.clamp(nodes_sum[:, 0:defines.NUM_CHECKED_VALUES], 0)
                    nodes_sum_l = int_loss(nodes_sum, b) * defines.INTER_MULT
                    l += nodes_sum_l
                    
                    l_mid += nodes_pred_l.item() + nodes_sum_l.item()
                
                
                self.optimizer_gnn.zero_grad()
                    
                l_g.backward()
                self.optimizer_gnn.step()
                l_sum += l_l.item() + l_g.item()
                l_sum1 += l_l.item()
                l_sum2 += l_g.item() if train_gnn and defines.GNN_MULT > 0 else 0
                #n += size_of_batch
                #n += labeled_edges.sum().item()
                n+=1
                
                seen_instances += size_of_batch
                
                if train_gnn and seen_instances > defines.GNN_INSTANCES_PER_EPOCH:
                    break
            
            val_loss, v_loss_l, v_loss_g = self.wrapper.evaluate_model(self.model, self.loss, self.x_val, self.y_val, defines.BATCH_SIZE, self.dglgraph.num_edges(), self.num_real_edges, train_gnn, True)
            
            print("epoch", epoch, ", train loss:", l_sum / n, "(" , l_sum1/n, ",", l_sum2/n, ",", l_mid/n, ") val loss:", 
                  val_loss, '(', v_loss_l, ",", v_loss_g, ')')            
            
            # Si la pérdida sobre training set ha aumentado un 20% entre epochs, reducimos el lr
            if val_loss > min_val_loss and last_epoch_loss != -1 and (last_epoch_loss * 1.2) < (l_sum / n):
                pass
                #self.scheduler.step()
                #print('reducing lr to', self.scheduler.get_last_lr(), "and loading saved model")
                #self.wrapper.load_model(self.model)
                #last_val_epoch = epoch
                
                #val_loss_load = self.wrapper.evaluate_model(self.model, self.loss, self.x_val, self.y_val, defines.BATCH_SIZE, self.dglgraph.num_edges(), self.num_real_edges, train_gnn)
                #print("loaded model", "validation loss:", val_loss_load)
            else:
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    self.wrapper.save_model(self.model)
                    last_val_epoch = epoch
                    # if train_gnn and epoch % 10 == 0:
                    #     last_val_epoch = epoch
                    #     self.scheduler.step()
                    #     print('reducing lr to', self.scheduler.get_last_lr(), "because of ten epochs module")
                # elif (epoch - last_val_epoch > 20) or (train_gnn and (epoch - last_val_epoch > 50)):
                #     last_val_epoch = epoch
                #     self.scheduler.step()
                #     print('reducing lr to', self.scheduler.get_last_lr(), "because of validation loss")
                      
                last_epoch_loss = l_sum / n
            
            if (not train_gnn) and epoch >= defines.EPOCHS_WITHOUT_GNN:
                train_gnn = True
                self.model.set_training_gnn(train_gnn)
                self.optimizer = self.wrapper.generate_optimizer(self.model.parameters(), lr=defines.LEARNING_RATE_GNN)
                self.scheduler = self.wrapper.generate_scheduler(self.optimizer,
                                            step_size=defines.DECAY_STEP,
                                            gamma=defines.GAMMA)
                min_val_loss = np.inf
                last_epoch_loss = -1
                print('GNN activated')
                
            if False and train_gnn and epoch % 30 == 0:
                defines.NUM_GNN_TO_TRAIN = min(defines.NUM_GNN_TO_TRAIN + 1, 5)
                print('Activada capa GNN:', defines.NUM_GNN_TO_TRAIN)
                min_val_loss = np.inf
                last_epoch_loss = -1

        x_test = self.wrapper.create_tensor(self.dataset.x_test)
        y_test = self.wrapper.create_tensor(self.dataset.y_test)

        test_loss = self.wrapper.evaluate_model(self.model, self.loss, x_test, y_test, defines.BATCH_SIZE, self.dglgraph.num_edges(), self.num_real_edges, train_gnn)
        print("test loss:", test_loss)
              
        
    def PointIndicesToNull(self):
        tfcId = self.dglgraph.edata['tfcId']
        return np.mod(list(range(len(tfcId))), 10)==0 
        
    def draw_test_data(self):
        self.model.eval()
        
        with torch.no_grad():
            self.loss = TrafficLoss(out_limits_factor=defines.OUT_LIMITS_FACTOR)
            val_loss = self.wrapper.evaluate_model(self.model, self.loss, self.x_val, self.y_val, defines.BATCH_SIZE, self.dglgraph.num_edges(), self.num_real_edges, True)
            print("loaded model", "validation loss:", val_loss)
            
            tfcId = self.dglgraph.edata['tfcId']
            
            base = tfcId.clone()
            pointsToNull = self.PointIndicesToNull()
            base[tfcId==0] = -1
            base[tfcId>0] = -1
            print_map(self.map.graph, base, 'points', 0, original=True, print_points=True)
            
            
            measured = tfcId.clone()
            measured[tfcId==0] = 55
            measured[tfcId>0] = 1
            print_map(self.map.graph, measured, 'measured', 0)
            
            
            voided = tfcId.clone()
            pointsToNull = self.PointIndicesToNull()
            voided[~pointsToNull] = 1
            voided[pointsToNull] = 30
            voided[tfcId==0] = -1
            print_map(self.map.graph, voided, 'voided', 0)
            
            
            x_test = self.dataset.x_test
            y_test = self.dataset.y_test
            
            
            #res = {_idx:(_y>=0).sum() for _idx, _y in enumerate(y_test)}
            #sort_orders = sorted(res.items(), key=lambda x: x[1], reverse=True)
            
            
            #idxlist = [45, 609, 641, 730, 328, 691, 53, 145, 337, 366]
            #idxlist = [0, 10, 20]
            #idxlist = [10]
            #idxlist = range(20,40)
            idxlist = [366, 35]
            
            for idx in idxlist:
                x = x_test[idx]
                y = y_test[idx]
                
                date_pattern = '{:04d}{:02d}{:02d}_{:02d}:{:02d}'
                date_str = date_pattern.format(round(x[8] * defines.MAX_YEAR_VALUE),
                                               round(x[7] * defines.MAX_MONTH_VALUE),
                                               round(x[6] * defines.MAX_DAY_VALUE),
                                               round(x[9] * defines.MAX_HOUR_VALUE),
                                               round(x[10] * defines.MAX_MINUTE_VALUE))
                        
                loads = y[:, 0]
                
                print_map(self.map.graph, loads, date_str + '_a', idx)
                
                x_tensor = self.wrapper.create_tensor([x])
                y_model = self.model(x_tensor)
                
                labeled_edges = y>=0
                y_linear = y_model[0].view(1, self.dglgraph.num_edges(), -1)
                y_linear = y_linear[0, :, 0:1]
                y_linear[~labeled_edges] = -1
                print_map(self.map.graph, y_linear[:, 0], date_str + '_b', idx)
                
                y_gnn = y_model[1].view(1, self.dglgraph.num_edges(), -1)
                print_map(self.map.graph, y_gnn[0, :, 0], date_str + '_c', idx, original=False)
                
                
                diff = y_gnn[0, :, 0].cpu() - loads
                diff = np.abs(diff)
                diff[~(labeled_edges.flatten())] = -1
                print_map(self.map.graph, diff, date_str + '_x', idx, original=False, print_points=False, second_color_scale=True)
        
                diffCalc = y_gnn[0, :, 0].cpu() - loads
                diffCalc = diffCalc[(labeled_edges.flatten())]
                
                neg4 = diffCalc[diffCalc<-30]
                neg3 = diffCalc[(diffCalc > -30) & (diffCalc<-20)]
                neg2 = diffCalc[(diffCalc > -20) & (diffCalc<-10)]
                neg1 = diffCalc[(diffCalc > -10) & (diffCalc<0)]
                
                
                pos1 = diffCalc[(diffCalc <= 10) & (diffCalc>=0)]
                pos2 = diffCalc[(diffCalc <= 20) & (diffCalc>10)]
                pos3 = diffCalc[(diffCalc <= 30) & (diffCalc>20)]
                pos4 = diffCalc[diffCalc>30]
                
                print('total arcos: ' + str(len(diffCalc)), 'neg4:' + str(len(neg4)) + ', neg3:' + str(len(neg3)) + ', neg2:' + str(len(neg2)) + ', neg1:' + str(len(neg1)))
                print('pos1:' + str(len(pos1)) + ', pos2:' + str(len(pos2)) + ', pos3:' + str(len(pos3)) + ', pos4:' + str(len(pos4)))
                
                x = 0
                