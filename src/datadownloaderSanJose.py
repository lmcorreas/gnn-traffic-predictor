
from genericpath import isfile
import requests
import os
from os.path import isfile, join
import pandas as pd
from zipfile import ZipFile

import numpy as np

import defines
from datadownloader import DataDownloader, PROCESSED_DIR, RAW_DIR

class DataDownloaderSanJose(DataDownloader):
        
    def DownloadData(self):
        pass        

    def GetFilesList(self, dir=PROCESSED_DIR):
        files = [f for f in os.listdir(dir) 
                if isfile(join(dir, f)) and f.endswith('.csv') and (not f.startswith('.'))]
        
        read_files = [join(dir, f) for f in files]
        
        return read_files


    def __ProcessFile__(self, filePath, id):
        data = pd.read_csv(filePath, ';')
        
        data.drop(['tipo_elem', 'error', 'periodo_integracion'], axis='columns', inplace=True)
        data['fecha'] = pd.to_datetime(data['fecha'])        
        
        # Si disponemos de listado de identificadores, excluimos el resto al no ser de utilidad
        if(len(self.validNodeIds) > 0):
            data = data[data['id'].isin(self.validNodeIds)]
        
        data.to_csv(PROCESSED_DIR + str(id) + '.csv', index=False, date_format='%Y-%m-%d %H:%M')
        
        os.remove(RAW_DIR + str(id) + '.zip')
        os.remove(RAW_DIR + str(id) + '.csv')
        