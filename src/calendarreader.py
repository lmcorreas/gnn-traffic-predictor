from time import strftime
from datetime import date

import numpy as np
import pandas as pd

HOLIDAY_STR = 'festivo'
SATURDAY = 'sábado'
SUNDAY = 'domingo'
WORKDAY = 'laborable'

class CalendarReader():
    def __init__(self):
        self.filePath = './data/calendar/calendario.csv'
        self.__read_calendar__()
    
    def __read_calendar__(self):
        self.data = pd.read_csv(self.filePath, sep=';', skip_blank_lines=True,
                           usecols=[0,1,2], names=['fecha', 'dia_semana', 'tipo_dia'],
                           header=None, skiprows=1)

        self.data.dropna(subset=['fecha'], inplace=True)
        
        #self.data['fecha'] = pd.to_datetime(self.data['fecha'], format='%d/%m/%Y')
        
    
    def get_data_value(self, dateVal):
        v = [0]
        
        dateStr = pd.to_datetime(dateVal).strftime('%d/%m/%Y')
        
        r = self.data[self.data['fecha'] == dateStr]
        
        if(len(r) == 1):
            val1 = r['dia_semana'].iloc[0]
            val2 = r['tipo_dia'].iloc[0]
            if((not pd.isna(val2)) and val2.lower() == HOLIDAY_STR):
                v[0] = 1
            else:   # Laborable
                pass
        else:
            print('OJO!!! Datos inconsistentes!!! ', date)
        
        return v