from copy import copy
import os
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point

from networkx.classes.multidigraph import MultiDiGraph

import defines

from trafficControlPoints import TrafficControlPoints as tfc
from trafficControlPointsSanJose import TrafficControlPointsSanJose as tfcsj

highway_include_complete = {
    'motorway',
    'trunk',
    'primary',
    'secondary',
    'tertiary',
    'motorway_link',
    'trunk_link',
    'primary_link',
    'secondary_link',
    'tertiary_link'
}

highway_include_simplified = {
    'motorway',
    'trunk',
    'primary',
    'motorway_link',
    'trunk_link',
    'primary_link',
} 

highway_include = highway_include_complete
if defines.SECOND_MAP:
    highway_include = highway_include_simplified

class Map():
        
    def load_map(self):
        ox.config(use_cache=True, log_console=True)
        
        if defines.SECOND_MAP:
            filePath = defines.DATA_BASE_PATH + 'sanjose_simple.gml'
            filePathFull = defines.DATA_BASE_PATH + 'sanjose_full.gml'
        else:
            filePath = defines.DATA_BASE_PATH + 'madrid_simple.gml'
            filePathFull = defines.DATA_BASE_PATH + 'madrid_full.gml'
        
        if(not (os.path.exists(filePath) and os.path.exists(filePathFull))):
                        
            self.read_map()
                                   
            ox.io.save_graphml(self.G, filePathFull)
            
            self.simplify_map()
            
            ox.io.save_graphml(self.graph, filePath)
        
        self.graph = ox.io.load_graphml(filePath)
        self.fullGraph = ox.io.load_graphml(filePathFull)
                    
        return self.graph, self.fullGraph
    
    def read_map(self):
        hw_filter = '|'.join(highway_include)
        filter = '["highway"~"' + hw_filter +'"]'
            
        if defines.MAP_FROM_PLACE:
            self.G = ox.graph_from_place(defines.MAP_PLACE,
                                            network_type=defines.MAP_TYPE,
                                            simplify=False,
                                            custom_filter=filter
                                            )
        else:
            if defines.SECOND_MAP:
                self.G = ox.graph_from_bbox(37.4421, 37.249, -121.822, -122.1121,
                                        network_type=defines.MAP_TYPE,
                                            simplify=False,
                                            custom_filter=filter)
            else:
                self.G = ox.graph_from_bbox(40.4856, 40.3787, -3.6533, -3.7563,
                                        network_type=defines.MAP_TYPE,
                                            simplify=False,
                                            custom_filter=filter)
            #polygon = wkt.loads('POLYGON ((40.5268 -3.6888, 40.4769 -3.7581, 40.4331 -3.8423, 40.3943 -3.8378, 40.3591 -3.7927, 40.3561 -3.6784, 40.4083 -3.5905, 40.4454 -3.586, 40.485 -3.6353, 40.5118 -3.6506, 40.5268 -3.6888))')
            #self.G = ox.graph_from_polygon(polygon,
            #                                network_type=defines.MAP_TYPE,
            #                                simplify=False,
            #                                custom_filter=filter
            #                                )
            
            
    def remove_edges_without_valid_highway(self):
        edges_to_remove = [(s, d) for s, d, meta in self.G.edges.data() if
                        not isinstance(meta['highway'], list) and meta['highway'] not in highway_include]
        self.remove_edges(edges_to_remove)
    
    def remove_edges(self, edges_to_remove):
        for (s, d) in edges_to_remove:
            while self.G.has_edge(s, d):
                self.G.remove_edge(s, d)
                
                
    def simplify_map(self):        
        self.remove_edges_without_valid_highway()
        self.G = ox.add_edge_speeds(self.G, fallback=50.0)
        self.G = ox.utils_graph.remove_isolated_nodes(self.G)
                
        self.G = ox.simplify_graph(self.G)
                
        self.G = self.G.to_directed()
                
        self.G = ox.project_graph(self.G)
                
        self.G = ox.consolidate_intersections(self.G, tolerance=10, rebuild_graph=True)
        
        self.G = ox.add_edge_speeds(self.G, fallback=50.0)
        self.regularize_max_speed()
        self.G = ox.add_edge_travel_times(self.G)
        
        for e in self.G.edges.data():
            if e[2]['oneway'] == None:
                print('OJO! Hay arcos no dirigidos, podría ser problemático')
                

        # Eliminamos arcos repetidos.
        self.graph = MultiDiGraph(ox.get_digraph(self.G))
        
        self.simplify_lanes()
        

        largest_component = max(nx.weakly_connected_components(self.graph), key=len)
        self.graph = self.graph.subgraph(largest_component)
        
    def preprocess_graph(self):
        self.save_edge_idx()
        self.save_degrees()
        self.save_lanes()
        self.save_node_lanes()
        self.complete_coordinates()
    
    def save_edge_idx(self):
        for idx,data in enumerate(self.graph.edges.data()):
            s,d,_ = data
            self.graph[s][d][0]['idx'] = idx
    
    def save_degrees(self):
        for s,d,meta in self.graph.edges.data():
            for i in range(len(self.graph[s][d])):
                # Almacenamos para cada arco el grado de los nodos origen y destino, tanto entrada como salida
                self.graph[s][d][i]['origin_in_degree'] = self.graph.in_degree(s)
                self.graph[s][d][i]['origin_out_degree'] = self.graph.out_degree(s)
                self.graph[s][d][i]['dest_in_degree'] = self.graph.in_degree(d)
                self.graph[s][d][i]['dest_out_degree'] = self.graph.out_degree(d)
                    
    def save_lanes(self):
        for s,d,meta in self.graph.edges.data():
            if(len(self.graph[s][d]) > 1):
                print('a!!')
            
            for i in range(len(self.graph[s][d])):                
                if 'lanes' not in self.graph[s][d][i]:
                    self.graph[s][d][i]['lanes'] = 1

                self.graph[s][d][i]['lanes'] = int(self.graph[s][d][i]['lanes'])
                
    def save_node_lanes(self):
        numNodes = self.graph.number_of_nodes()
        
        arrInput = np.zeros(numNodes)
        arrOutput = np.zeros(numNodes)
        
        for s,d,meta in self.graph.edges.data():
            for i in range(len(self.graph[s][d])):
                lanes = self.graph[s][d][i]['lanes']
                
                arrInput[d] = arrInput[d] + lanes
                arrOutput[s] = arrOutput[s] + lanes
                
        for n, meta in self.graph.nodes.data():
            self.graph.nodes[n]['inputLanes'] = int(arrInput[n])
            self.graph.nodes[n]['outputLanes'] = int(arrOutput[n])
    
    def complete_coordinates(self):
        
        minLat = np.inf
        minLon = np.inf
        
        maxLat = -(np.inf)
        maxLon = -(np.inf)
        
        for n, meta in self.graph.nodes.data():
            x = self.graph.nodes[n]['x']
            y = self.graph.nodes[n]['y']
            
            points_list = [Point((x, y))]
        
            points = gpd.GeoSeries(points_list, crs=self.graph.graph['crs'])
            points_proj = points.to_crs('epsg:4326')
            
            self.graph.nodes[n]['lon_orig'] = points_proj[0].x
            self.graph.nodes[n]['lat_orig'] = points_proj[0].y
            
            minLon = min(minLon, points_proj[0].x)
            maxLon = max(maxLon, points_proj[0].x)
            
            minLat = min(minLat, points_proj[0].y)
            maxLat = max(maxLat, points_proj[0].y)
            
        for n, meta in self.graph.nodes.data():
            lon = self.graph.nodes[n]['lon_orig']
            lat = self.graph.nodes[n]['lat_orig']
            
            # Realizamos un escalado MinMax
            self.graph.nodes[n]['lon'] = (lon-minLon)/(maxLon-minLon)
            self.graph.nodes[n]['lat'] = (lat-minLat)/(maxLat-minLat)
    
    def simplify_lanes(self):
        for s,d,meta in self.graph.edges.data():
            for i in range(len(self.graph[s][d])):
                if type(self.graph[s][d][i]['lanes']) is list:
                    self.graph[s][d][i]['lanes'] = str(np.sum(np.array(self.graph[s][d][i]['lanes']).astype(np.int)))
                    
        for s,d,meta in self.graph.edges.data():
            for i in range(len(self.graph[s][d])):
                if type(self.graph[s][d][i]['lanes']) is list:
                    a=0
        
    def regularize_max_speed(self):
        highway_maxspeed = {'motorway': 120.0, 'trunk': 80.0, 'primary': 80.0, 'secondary': 70.0, 'tertiary':50.0, 'unclassified':50.0, 'residential':30.0,
        'motorway_link':80.0, 'trunk_link':80.0, 'primary_link':80.0, 'secondary_link':70.0, 'tertiary_link':50.0, 'living_street':30.0, 'service':30.0, 'road':50.0}
                
        self.simplify_highway()
        
        #self.G._clean_maxspeed
        
        for s,d,meta in self.G.edges.data():
            for i in range(len(self.G[s][d])):
                if 'maxspeed' not in self.G[s][d][i]:
                    if isinstance(self.G[s][d][i]['highway'], list):
                        self.G[s][d][i]['highway'] = self.G[s][d][i]['highway'][0]
                    
                    self.G[s][d][i]['maxspeed'] = str(highway_maxspeed[self.G[s][d][i]['highway']])
                elif type(self.G[s][d][i]['maxspeed']) is list:
                    for j in range(len(self.G[s][d][i]['maxspeed'])):
                        if(self.G[s][d][i]['maxspeed'][j]=='ES:motorway'):
                            self.G[s][d][i]['maxspeed'][j] = '120'
                        elif '|' in self.G[s][d][i]['maxspeed'][j]:
                            self.G[s][d][i]['maxspeed'][j] = str(np.min(np.array(self.G[s][d][i]['maxspeed'][j].split('|')).astype(np.float)))
                        elif 'km/h' in self.G[s][d][i]['maxspeed'][j]:
                            idx = self.G[s][d][i]['maxspeed'][j].index('km/h')
                            self.G[s][d][i]['maxspeed'][j] = self.G[s][d][i]['maxspeed'][j][0:idx]
                        elif 'mph' in self.G[s][d][i]['maxspeed'][j]:
                            # En el caso raro que venga con mph, aprovechamos a convertirlo
                            idx = self.G[s][d][i]['maxspeed'][j].index('mph')
                            self.G[s][d][i]['maxspeed'][j] = str(float(self.G[s][d][i]['maxspeed'][j][0:idx]) * 1.6093)
                            
                    self.G[s][d][i]['maxspeed'] = str(np.min(np.array(self.G[s][d][i]['maxspeed']).astype(np.float)))
                        
                if 'speed_kph' not in self.G[s][d][i]:
                    self.G[s][d][i]['speed_kph'] = highway_maxspeed[self.G[s][d][i]['highway']]
                elif type(self.G[s][d][i]['speed_kph']) is list:
                    self.G[s][d][i]['speed_kph'] = np.min(np.array(self.G[s][d][i]['speed_kph']).astype(np.float))
                elif self.G[s][d][i]['speed_kph'] > 120.0:
                    self.G[s][d][i]['speed_kph'] = highway_maxspeed[self.G[s][d][i]['highway']]
                    
                if 'lanes' not in self.G[s][d][i]:
                    self.G[s][d][i]['lanes'] = 1

    def simplify_highway(self):
        for s, d, meta in self.G.edges.data():
            if isinstance(meta['highway'], list):
                first = next(filter(lambda hwy: hwy in highway_include, meta['highway']), None)
                self.G[s][d][0]['highway'] = first
                
    def check_double_edges(self):
        for s,d,meta in self.G.edges.data():
            if(len(self.G[s][d]) > 1):
                print('a!!')
                
    def get_edge_from_latlon(self, lat, lon):
        points_list = [Point((lon, lat))]
        points = gpd.GeoSeries(points_list, crs='epsg:4326')
        points_proj = points.to_crs(self.graph.graph['crs'])
        
        XList = [pt.x for pt in points_proj]
        YList = [pt.y for pt in points_proj]
        
        ne = ox.nearest_edges(self.graph, XList, YList, return_dist=True)
        points = gpd.GeoSeries(points_list)
        
        values = zip(ne[0], ne[1], points_proj)        
        rawList = list(values)
        dataList = sorted(rawList, key = lambda x: x[1])
        
        for x in dataList:
            if(x[1]<defines.DISTANCE_THRESHOLD):                
                vFrom = x[0][0]
                vTo = x[0][1]
                
                return vFrom, vTo
            
        return -1, -1
    
    def gen_traffic_control_point_reader(self):
        if defines.SECOND_MAP:
            return tfcsj()
        
        return tfc()
                
    def filter_traffic_control_points(self):
        for s,d,meta in self.graph.edges.data():
            for i in range(len(self.graph[s][d])):
                self.graph[s][d][i]['tfcId'] = 0
        
        tf = self.gen_traffic_control_point_reader()

        idsList, longList, latList = tf.returnLongLatList()

        points_list = [Point((lng, lat)) for lng, lat  in zip(longList, latList)]
        
        points = gpd.GeoSeries(points_list, crs='epsg:4326')
        points_proj = points.to_crs(self.graph.graph['crs'])
        
        XList = [pt.x for pt in points_proj]
        YList = [pt.y for pt in points_proj]
            
        ne = ox.nearest_edges(self.graph, XList, YList, return_dist=True)
        points = gpd.GeoSeries(points_list)
        
        self.pointIds = []
        self.pointsToDraw = []
        nearne = []
        
        values = zip(ne[0], ne[1], points_proj, idsList)
        rawList = list(values)
        dataList = sorted(rawList, key = lambda x: x[1])
        
        
        for x in dataList:            
            if(x[1]<defines.DISTANCE_THRESHOLD):
                nearne.append(x[0])
                
                vFrom = x[0][0]
                vTo = x[0][1]
                
                ed = self.graph.get_edge_data(vFrom, vTo)
                if ed is not None and self.graph.edges[vFrom,vTo,0]['tfcId'] == 0:
                    self.pointsToDraw.append({"x": x[2].x, "y": x[2].y, "O": x[0][0], "D": x[0][1]})
                    self.pointIds.append(x[3])
                    self.graph.edges[vFrom,vTo,0]['tfcId']=x[3]
        
        self.pointIds.sort()
        
        return self.pointIds