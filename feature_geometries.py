import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
from shapely import wkt
import itertools

def remove_doubleID_streets(df):
    out_df = df.copy()
    for i, val in enumerate(out_df.highway):
        if isinstance(val, list):
            out_df.drop(i, inplace=True)
    return(out_df)


def distance_to_roadway(gps, roadway):
    dists = []
    for i in roadway.geometry:
        dists.append(i.distance(gps))
    return(np.min(dists))


def find_closest_road(gps, roads, buffer_dist = 0.0003):

    road_index = roads.sindex
    circle = gps.buffer(buffer_dist) #build buffer around point (~ 30 meters)
    
    possible_matches_index = list(road_index.intersection(circle.bounds)) #get index of possible nearest roads
    possible_matches = roads.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(circle)].copy()

    #get distances to roads in buffer
    precise_matches['distance'] = precise_matches['geometry'].distance(gps)
    
    if precise_matches['distance'].empty is False:
        return(precise_matches.sort_values(['distance']).drop_duplicates('distance').iloc[0, 3])
    else:
        return('outside_area')


def nearest_intersection(gps, intersections):
    closest_point = nearest_points(gps, MultiPoint(intersections.values))[1]
    return(gps.distance(closest_point))


def distance_to_zoning(gps, zone):
    dists = []
    for i in zone.geometry:
        dists.append(i.distance(gps))
    return(np.min(dists))


def import_gpd(filename):
    data = pd.read_csv(filename)
    data['geometry'] = data['geometry'].apply(wkt.loads)
    data_gpd = gpd.GeoDataFrame(data, geometry = data['geometry'], crs={'init' :'epsg:4326'})
    data_gpd = data_gpd.drop(['Unnamed: 0'], axis = 1)
    return(data_gpd)


def clean_roads(df):
    # Cleaning road categories ...
    df['road_type'] = df['road_type'].str.replace('_link', '')
    df['road_type'] = np.where(df['road_type'] == 'trunk', 'secondary', df['road_type'])
    df['road_type'] = np.where(df['road_type'] == 'living_street', 'residential', df['road_type'])
    df['road_type'] = np.where(df['road_type'] == 'a', 'unclassified', df['road_type'])

    # Drop all point not measured on a OpenStreetMap roadway
    df = df[df.road_type != 'outside_area']

    return(df)
