import pandas as pd

from trafficControlPoints import TrafficControlPoints, ControlPoint

class TrafficControlPointsSanJose(TrafficControlPoints):
    
    def readFile(self):
        self.df = pd.read_csv(self.filename, sep=',', quotechar='"', header=None)

        self.points = []
        for i, p in self.df.iterrows():
            self.points.append(ControlPoint(p[0], '', '', '', p[2], p[1]))

    def returnLongLatList(self):
        if(len(self.points) == 0):
            self.readFile()
        return list(self.df[0]), list(self.df[2]), list(self.df[1])
