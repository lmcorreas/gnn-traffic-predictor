# !/usr/bin/env python

import json
from flask import Flask, request, abort

from app import App

webapp = Flask(__name__)
app = App()

@webapp.route('/traffic/point', methods=['GET'])
def get_point_traffic_info():
    args = request.args
    
    lat = args.get('lat')
    lon = args.get('lon')
    
    year = args.get('year')
    month = args.get('month')
    day = args.get('day')
    hour = args.get('hour')
    minute = args.get('minute')
    
    if None not in (lat, lon):
        info = app.get_point_traffic_info(lat, lon, year, month, day, hour, minute)
        
        return json.dumps(info), 200, {'Content-Type': 'application/json'}
    else:
        info = {'message': 'Verify if latitude and longitude are correct'}
        return json.dumps(info), 400, {'Content-Type': 'application/json'}
        
@webapp.route('/traffic/info', methods=['GET'])
def get_traffic_info():
    args = request.args
    
    year = args.get('year')
    month = args.get('month')
    day = args.get('day')
    hour = args.get('hour')
    minute = args.get('minute')
    
    info = app.get_traffic_info(year, month, day, hour, minute)
    
    return json.dumps(info), 200, {'Content-Type': 'application/json'}
    
if __name__ == '__main__': 
    app.init_app()
    webapp.run(host='0.0.0.0', debug=True, use_reloader=False)
    print('APP is up and kicking!')