
from genericpath import isfile
import requests
import os
from os.path import isfile, join
import pandas as pd
from zipfile import ZipFile

import numpy as np

import defines

MASKED_URL = 'https://datos.madrid.es/egob/catalogo/208627-[_ID_]-transporte-ptomedida-historico.zip'
RAW_DIR = './' + defines.DATA_BASE_PATH + 'raw/'
PROCESSED_DIR = './' + defines.DATA_BASE_PATH + 'processed/'
DATASET_DIR = './' + defines.DATA_BASE_PATH + 'dataset'

X_TRAIN_FILE = '/xtrain.npy'
Y_TRAIN_FILE = '/ytrain.npy'
X_VAL_FILE = '/xval.npy'
Y_VAL_FILE = '/yval.npy'
X_TEST_FILE = '/xtest.npy'
Y_TEST_FILE = '/ytest.npy'

class DataDownloader():
    
    def __init__(self, forceDownload=False, validNodeIds=[]):
        self.validNodeIds = validNodeIds
        self.forceDownload = forceDownload
    
    def DownloadData(self):
        #files = self.GetFilesList(RAW_DIR)
        #for f in files:
        #    self.__ProcessFile__(f, 103)
        
        for id in range(defines.MIN_ID, defines.MAX_ID+1):
            filePath = PROCESSED_DIR + str(id) + '.csv'
            
            if(self.forceDownload or (not os.path.exists(filePath))):
                csvFileName = self.__DownloadRawData__(id)
                self.__ProcessFile__(csvFileName, id)
                
                
    def __DownloadRawData__(self, id):
        
        print('Downloading file with id', id)
    
        url = MASKED_URL.replace('[_ID_]', str(id))
        
        r = requests.get(url)
        
        zipFileName = RAW_DIR + str(id) + '.zip'
        
        open(zipFileName, 'wb').write(r.content)
        
        with ZipFile(zipFileName, 'r') as zip:
            origFileName = zip.filelist[0].filename
            
            zip.extract(origFileName, RAW_DIR)
        
        csvFileName = RAW_DIR + str(id) + '.csv'
        os.rename(RAW_DIR + origFileName, csvFileName)
        
        print('File with id downloaded', id)
        
        return csvFileName
        

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
        
    
    def GetFilesList(self, dir=PROCESSED_DIR):
        files = [f for f in os.listdir(dir) 
                    if isfile(join(dir, f)) and f.endswith('.csv') and (not f.startswith('.'))]
        
        sorted_files = sorted(files, key=(lambda x: int(x.split('.')[0])))
        
        read_files = [join(dir, f) for f in sorted_files]
        
        return read_files


    def SaveDataset(self, x_train, y_train, x_val, y_val, x_test, y_test):
        np.save(DATASET_DIR + X_TRAIN_FILE, x_train)
        np.save(DATASET_DIR + Y_TRAIN_FILE, y_train)
        np.save(DATASET_DIR + X_VAL_FILE, x_val)
        np.save(DATASET_DIR + Y_VAL_FILE, y_val)
        np.save(DATASET_DIR + X_TEST_FILE, x_test)
        np.save(DATASET_DIR + Y_TEST_FILE, y_test)
        
    def LoadDataset(self):
        return (
            np.load(DATASET_DIR + X_TRAIN_FILE),
            np.load(DATASET_DIR + Y_TRAIN_FILE),
            np.load(DATASET_DIR + X_VAL_FILE),
            np.load(DATASET_DIR + Y_VAL_FILE),
            np.load(DATASET_DIR + X_TEST_FILE),
            np.load(DATASET_DIR + Y_TEST_FILE)
        )
        
    def ExistsDataset(self):
        return (
                os.path.exists(DATASET_DIR + X_TRAIN_FILE)
                and os.path.exists(DATASET_DIR + Y_TRAIN_FILE)
                and os.path.exists(DATASET_DIR + X_VAL_FILE)
                and os.path.exists(DATASET_DIR + Y_VAL_FILE)
                and os.path.exists(DATASET_DIR + X_TEST_FILE)
                and os.path.exists(DATASET_DIR + Y_TEST_FILE)
        )