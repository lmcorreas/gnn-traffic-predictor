import pandas as pd

import defines

class ControlPoint():
    def __init__(self, id, code, utm_x, utm_y, long, lat):
        self.id = id
        self.code = code
        self.utm_x = utm_x
        self.utm_y = utm_y
        self.long = long
        self.lat = lat

class TrafficControlPoints():
    
    def __init__(self):
        self.points = []
        self.filename = defines.DATA_BASE_PATH + defines.SENSOR_INFO_FILE_NAME
    
    def getPoints(self):
        if(len(self.points) == 0):
            self.readFile()
            
        return self.points
    
    def readFile(self):
        
        self.df = pd.read_csv(self.filename, sep=';', quotechar='"')
        self.df = self.df.drop(['nombre', 'distrito', 'tipo_elem'], axis=1)
        #print(self.df.head(5))

        self.points = []
        for i, p in self.df.iterrows():
            self.points.append(ControlPoint(p.id, p.cod_cent, p.utm_x, p.utm_y, p.longitud, p.latitud))

    def returnLongLatList(self):
        if(len(self.points) == 0):
            self.readFile()
        return list(self.df['id']), list(self.df['longitud']), list(self.df['latitud'])
