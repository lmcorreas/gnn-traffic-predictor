from dgl.data import DGLDataset
import pandas as pd
import numpy as np
import math

from datadownloader import DataDownloader
from datadownloaderSanJose import DataDownloaderSanJose
from calendarreader import CalendarReader

from datetime import datetime

import defines

class TrafficDataset(DGLDataset):
    def __init__(self, graphTfcData, listOfNodes=[]):
        self.listOfNodes = np.sort(listOfNodes)
        self.graphTfcData = graphTfcData
        
        self.data_type = np.float32
        if defines.USE_DOUBLE:
            self.data_type = np.double
        
        self.tfcToIdx = {}
        self.idxToTfc = {}
        for idx, n in enumerate(self.graphTfcData):
            v = int(n)
            if v>0:
                self.tfcToIdx[v] = idx
                self.idxToTfc[idx] = v
                
        if defines.SECOND_MAP:
            self.dd = DataDownloaderSanJose(validNodeIds=self.listOfNodes)    
        else:            
            self.dd = DataDownloader(validNodeIds=self.listOfNodes)
        
        super().__init__(name='traffic')
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
    # Número de muestras de entrenamiento
    def __len__(self):
        return len(self.x_train)
    
    # Tamaño del dataset completo
    def size(self):
        return len(self.x)
    
    def has_cache(self):
        return self.dd.ExistsDataset()
    
    def save(self):
        self.dd.SaveDataset(self.x_train, self.y_train, self.x_val, self.y_val,
                            self.x_test, self.y_test)
        
    def load(self):
        data = self.dd.LoadDataset()
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        
        if defines.VOID_SOME_POINTS_TRAINING:
            self.x_train_copy = self.x_train.copy()
            self.y_train_copy = self.y_train.copy()
            self.x_val_copy = self.x_val.copy()
            self.y_val_copy = self.y_val.copy()
            self.x_test_copy = self.x_test.copy()
            self.y_test_copy = self.y_test.copy()
            
            self.x_train_copy = np.concatenate([self.x_train_copy, self.x_val_copy, self.x_test_copy])
            self.y_train_copy = np.concatenate([self.y_train_copy, self.y_val_copy, self.y_test_copy])
                        
    def process(self):        
        dataList = []
        labels = []
        
        if not defines.SECOND_MAP:
            self.calendarReader = CalendarReader()
        
        self.dd.DownloadData()
        
        self.files = self.dd.GetFilesList()
        
        for f in self.files:
            data = pd.read_csv(f)
            
            print('Processing file', f)
            
            x, y = self.processData(data)
            
            dataList.extend(x)
            labels.extend(y)
            
            print('File', f, 'processed')
            
        self.x = np.array(dataList)
        self.y = np.array(labels)
        
        if defines.EXPORT_PROCESSED_FILE:
            self.export_processed_csv()
        
        self.divideDataset()
        
    def export_processed_csv(self):
        
        f = open("data/ready/processed.csv", "w")
        f.write('id,lunes,martes,miercoles,jueves,viernes,sabado,dia,mes,anyo,hora,minuto,festivo,carga\n')
        
        for id, d in enumerate(zip(self.x, self.y)):
            x = d[0]
            y = d[1]
            
            for idx, z in enumerate(y):
                if z[0] >= 0:
                    line = '{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'
                    s = line.format(self.idxToTfc[idx], x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                                    x[7], x[8], x[9], x[10], x[11], z[0])
                    f.write(s)
                    
        f.close()
        
    def divideDataset(self):
        np.random.seed=defines.RANDOM_SEED
        
        n_samples = self.size()
        
        indices = np.random.permutation(n_samples)
        training_size = math.ceil(n_samples * defines.TRAINING_SIZE)
        val_size = math.ceil(n_samples * defines.VAL_SIZE)
        training_idx, val_idx, test_idx = indices[:training_size], indices[training_size:training_size+val_size], indices[training_size+val_size:]

        self.x_train, self.x_val, self.x_test = self.x[training_idx,:], self.x[val_idx,:], self.x[test_idx,:]
        self.y_train, self.y_val, self.y_test = self.y[training_idx], self.y[val_idx], self.y[test_idx]

        
    def processData(self, data):
        if defines.SECOND_MAP:
            return self.processDataSanJose(data)
        
        return self.processDataMadrid(data)
    
    def processDataMadrid(self, data):
        data = data[data['id'].isin(self.listOfNodes)]
        data = data.replace(np.nan, -1)
        
        # Convertimos la fecha a datetime
        data['fecha_dt'] = pd.to_datetime(data['fecha'])
        
        distinctDates = data['fecha_dt'].unique()
        distinctDates.sort()
    
        data['diasemana'] = data['fecha_dt'].dt.weekday
        data['dia'] = data['fecha_dt'].dt.day
        data['mes'] = data['fecha_dt'].dt.month
        data['anyo'] = data['fecha_dt'].dt.year
        data['hora'] = data['fecha_dt'].dt.hour
        data['minuto'] = data['fecha_dt'].dt.minute
        data.sort_values(by=['fecha_dt'], inplace=True)
        
        # Dim 1: fecha
        # Dim 2: característica
        xData = np.zeros((len(distinctDates), defines.NUM_INPUT_FEATURES), dtype=self.data_type)
        
        # Dim 1: fecha
        # Dim 2: nodo
        # Dim 3: característica
        yData = np.full((len(distinctDates), len(self.graphTfcData), defines.NUM_CHECKED_VALUES), -1, dtype=self.data_type)
        
        #f = open("control_202202.csv", "w")
        #f.write('fecha,total\n')
        
        for idx, d in enumerate(distinctDates):
            dateData = data[data['fecha_dt']==d]
            
            #line = "{},{}\n"
            #f.write(line.format(str(d),str(len(dateData))))
            
            #print(d, len(dateData))
            
            weekday = dateData.iloc[0]['diasemana']
            # Si es domingo, no marcamos ningún flag
            if weekday < 6:
                xData[idx,dateData.iloc[0]['diasemana']]=1
                
            xData[idx, 6] = dateData.iloc[0]['dia'] / defines.MAX_DAY_VALUE
            xData[idx, 7] = dateData.iloc[0]['mes'] / defines.MAX_MONTH_VALUE
            xData[idx, 8] = dateData.iloc[0]['anyo'] / defines.MAX_YEAR_VALUE
            xData[idx, 9] = dateData.iloc[0]['hora'] / defines.MAX_HOUR_VALUE
            xData[idx, 10] = dateData.iloc[0]['minuto'] / defines.MAX_MINUTE_VALUE
            
            calData = self.calendarReader.get_data_value(d)
            xData[idx,11]=calData[0]
            
            # Calculamos desde y hasta las 0:00
            xData[idx, 12] = ((xData[idx, 9]*60)+xData[idx, 10]) / defines.MAX_DAY_MINUTES_VALUE
            xData[idx, 13] = (1440 - (xData[idx, 12])) / defines.MAX_DAY_MINUTES_VALUE
            
            prevDt = d - np.timedelta64(1, 'D')
            calPrevData = self.calendarReader.get_data_value(prevDt)
            xData[idx,14]=calPrevData[0]
            
            nextDt = d + np.timedelta64(1, 'D')
            calNextData = self.calendarReader.get_data_value(nextDt)
            xData[idx,15]=calNextData[0]
            
            # Asignamos dummy meses
            monthIdx = int(15+xData[idx, 7])
            xData[idx, monthIdx] = 1
                        
            for row in dateData.itertuples(index=False):
                nIdx = self.tfcToIdx[row.id]
                yData[idx, nIdx, 0] = row.carga # Valor máximo es 100, por lo que lo dejamos tal cual
                #yData[idx, nIdx, 1] = (row.intensidad * 100) / defines.MAX_INTENSITY_VALUE
                #yData[idx, nIdx, 2] = row.ocupacion
                # Desestimado hasta nueva orden. Solo está informado para algunos puntos de medición
                #yData[idx,nIdx,3] = row.vmed

        #f.close()
        
        return xData, yData
    
    def processDataSanJose(self, data):
        
        # Dim 1: fecha
        # Dim 2: característica
        xData = np.zeros((len(data), defines.NUM_INPUT_FEATURES), dtype=self.data_type)
        
        # Dim 1: fecha
        # Dim 2: nodo
        # Dim 3: característica
        yData = np.full((len(data), len(self.graphTfcData), defines.NUM_CHECKED_VALUES), -1, dtype=self.data_type)
        
        dateIdx = 0
        for e in data.iterrows():
            #dt = datetime(e[1][0])
            
            dt = datetime.strptime(e[1][0], '%Y-%m-%d %H:%M:%S')
            
            weekday = dt.weekday()
            # Si es domingo, no marcamos ningún flag
            if weekday < 6:
                xData[dateIdx,weekday]=1
                
            xData[dateIdx, 6] = dt.day / defines.MAX_DAY_VALUE
            xData[dateIdx, 7] = dt.month / defines.MAX_MONTH_VALUE
            xData[dateIdx, 8] = dt.year / defines.MAX_YEAR_VALUE
            xData[dateIdx, 9] = dt.hour / defines.MAX_HOUR_VALUE
            xData[dateIdx, 10] = dt.minute / defines.MAX_MINUTE_VALUE
            
            # Calculamos desde y hasta las 0:00
            xData[dateIdx, 12] = ((dt.hour*60)+dt.second) / defines.MAX_DAY_MINUTES_VALUE
            xData[dateIdx, 13] = (1440 - (xData[dateIdx, 12])) / defines.MAX_DAY_MINUTES_VALUE
            
            # Asignamos dummy meses
            monthIdx = int(15+dt.month)
            xData[dateIdx, monthIdx] = 1
            
            for idx, val in e[1].items():
                # Si no es numérico, no se trata de in id de punto de control de tráfico
                if idx.isnumeric():
                    if int(idx) in self.tfcToIdx:
                        nIdx = self.tfcToIdx[int(idx)]
                        
                        yData[dateIdx, nIdx, 0] = int(val)
                
            #Pasamos al siguiente índice de fecha
            dateIdx = dateIdx + 1
                    
        return xData, yData