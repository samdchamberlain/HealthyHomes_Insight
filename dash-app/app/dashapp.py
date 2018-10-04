# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from dash.dependencies import Input, Output
from sklearn.externals import joblib
import copy

import geopandas as gpd
import pandas as pd
import itertools
import rasterio
from geopy.geocoders import GoogleV3
from shapely.geometry import Point
from feature_geometries import *

from app import app


# Define for IIS module registration.
wsgi_app = app.wsgi_app

# Connect dash to flask
app = dash.Dash(__name__, server=app, url_base_pathname='/')

# Import datasets for feature building
all_roads = import_gpd('data/all_roads.csv')
all_intersections = import_gpd('data/all_intersections.csv')
all_zoning = import_gpd('data/east_zoning.csv')
census_tracts = import_gpd('data/census_tracts.csv')
heatmap = import_gpd('data/heatmap.csv')
neighborhoods = import_gpd('data/neighborhood_summaries.csv')
region = import_gpd('data/region.csv')


# Import EDF-GSV data for visualizations
mapbox_access_token = 'pk.eyJ1Ijoic2NoYW1iZXJsYWluIiwiYSI6ImNqbWNpMGZvNTBmOHUzcHM0MmltMGZwZ3cifQ.a0tYS4RsPQfjr-R5bfJv5g'
GSV_df = import_gpd('data/GSV_all.csv')

# Mapping layout
layout = dict(
    autosize=True,
    height=800,
    font=dict(color='#fffcfc'),
    titlefont=dict(color='#fffcfc', size='18'),
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor="#191A1A",
    paper_bgcolor="#020202",
    legend=dict(font=dict(size=16), orientation='h'),
    title='Air quality at your address and neighborhood',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="dark",
        center=dict(
            lon=-122.271111,
            lat=37.804363
        ),
        zoom=13,
    )
)


def find_nearby_neighborhoods(location, neighborhoods, buffer_dist = 0.015):

    neighborhood_index = neighborhoods.sindex
    circle = location.buffer(buffer_dist) #build a km buffer around location
    
    possible_matches_index = list(neighborhood_index.intersection(circle.bounds)) #get index of possible neighborhoods
    possible_matches = neighborhoods.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(circle)].copy()

    return precise_matches


def get_healthy_suggestions(location, neighborhoods, price_range = 0.15, buffer_dist = 0.015):
    
    #find your current neighborhood...
    current_hood = neighborhoods[neighborhoods.intersects(location)]
    
    #find all neighborhoods within a mile (0.015 deg buffer)
    nearby_hoods = find_nearby_neighborhoods(location, neighborhoods)
    
    #subset neighborhoods in similar price bracket (default is +/- 20%)
    upper = 1 + price_range
    lower = 1 - price_range
    similar_hoods = nearby_hoods[nearby_hoods.price.between(current_hood['price'].iloc[0]*lower, 
                                                            current_hood['price'].iloc[0]*upper)]
    
    
    #find which neighborhoods have lower exposures 
    healthy_hoods = similar_hoods[similar_hoods.no2 < current_hood['no2'].iloc[0]]
    
    #how much lower are the exposure rates?
    healthy_hoods['percent_diff'] = np.ceil((1 - (healthy_hoods['no2']/current_hood['no2'].iloc[0]))*100) # round up to be conservative
    healthy_hoods['percent_diff'] = healthy_hoods['percent_diff'].astype(int)
    
    return healthy_hoods


def map_estimate(address_data, pollutant, lat, lon):

    address_layout = copy.deepcopy(layout)
    address_layout['mapbox']['center']['lat'] = lat
    address_layout['mapbox']['center']['lon'] = lon

    if pollutant.name == 'NO2':
        heat_pollutant = heatmap['no2']
    else:
        heat_pollutant = heatmap['bc']


    return {
        "data": [
                {
                    "type": "scattermapbox",
                    "lat": list(heatmap['Lat']),
                    "lon": list(heatmap['Long']),
                    "text": list(heat_pollutant),
                    "mode": "markers",
                    "name": "Neighborhood",
                    "marker": {
                        "size": 60,
                        "opacity": 0.01,
                        "color": heat_pollutant
                    }
                },

                {
                    "type": "scattermapbox",
                    "lat": list(address_data['Latitude']),
                    "lon": list(address_data['Longitude']),
                    "text": list(pollutant),
                    "mode": "markers",
                    "name": "Address",
                    "marker": {
                        "size": 20,
                        "opacity": 0.9,
                        "color": pollutant
                    }
                }

            ],
        "layout": address_layout
    }

app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501
app.layout = html.Div(
    [
        html.H1("How's the air quality at your East Bay home?", style={'textAlign': 'center'}),
    
        html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            options = [
                                {'label': 'Nitrogen dioxide', 'value': 'no2'},
                                {'label': 'Black carbon', 'value': 'bc'}
                            ],
                            value = 'no2',
                            id = 'dropdown-pollutant'
                        )
                    ],
                    className='six columns',
                    style={'textAlign': 'center', 'fontSize': 20}
                ),

                dcc.Input(id='input-box', type='text', value='4000 Telegraph Ave, Oakland'),
                html.Button('Submit', id='button', style={'textAlign': 'center', 'fontSize': 18}),
            
            ], style={'textAlign': 'center', 'fontSize': 24}),


        html.Div(
            [
                dcc.Graph(id='address_map',
                    style={'margin-top': '10'})
            ], className = "six columns"
        ),

        html.Div(
        	[
        		dcc.Markdown(id='output-container-button')
        	], style={'textAlign': 'center', 'fontSize': 36, 'margin-top': '65'}, className = 'five columns'
        ),

        html.Div(id='suggestions', children='',
            style={'textAlign': 'center', 'margin-top': '50'}, 
            className = "five columns"),

        # html.Div(id='container'),

        #html.Div(id='datatable', children=''),

        # dt.DataTable(
        #     rows=[{}], # initialise the rows
        #     row_selectable=True,
        #     filterable=True,
        #     sortable=True,
        #     selected_row_indices=[],
        #     id='datatable'
        # ),

        html.Div(id='intermediate_value', style={'display': 'none'})
    ]
)


@app.callback(
    dash.dependencies.Output('intermediate_value', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])

def find_exposure(n_clicks, value):
	
    ## A. Get the lat long of address input  by user
    address = str(value)
    geolocator = GoogleV3(api_key='AIzaSyBZLOJfP1yw-F5T26O_nNcjXpceL6KrD3Q')
    location = geolocator.geocode(address) # contains full address and lat long data
    geolocation = Point(location[1][1], location[1][0])
    
    ## B. Calculate features
    # Re-project in utm to calculate distances to input point below
    highway_utm = all_roads[all_roads.highway == 'motorway'].to_crs({'init': 'epsg:32610'}).copy()
    primary_utm = all_roads[all_roads.highway == 'primary'].to_crs({'init': 'epsg:32610'}).copy()
    secondary_utm = all_roads[all_roads.highway == 'secondary'].to_crs({'init': 'epsg:32610'}).copy()
    intersections_utm = all_intersections.to_crs({'init': 'epsg:32610'}).copy()
    traffic_signals = intersections_utm[intersections_utm.highway == 'traffic_signals']
    industry_utm = all_zoning[all_zoning.zone == 'industrial'].to_crs({'init': 'epsg:32610'}).copy()

    # 1. closest road_type
    road_type = find_closest_road(geolocation, all_roads)

    #store in geodataframe for reprojections needed to following features
    all_features = [{'geometry': geolocation, 'road_type': road_type}]
    features_df = pd.DataFrame(all_features)
    features_df = gpd.GeoDataFrame(features_df, geometry=features_df.geometry, crs={'init' :'epsg:4326'})
    features_df['Latitude'] = location[1][0]
    features_df['Longitude'] = location[1][1]

    # 2. Distance to dominant roadtypes and intersections
    # Re-project in utm to calculate distance to input point
    location_utm = features_df.to_crs({'init': 'epsg:32610'}).copy()

    # Calculate distances to major roadways
    features_df['closest_highway'] = distance_to_roadway(location_utm['geometry'][0], highway_utm)
    features_df['closest_primary'] = distance_to_roadway(location_utm['geometry'][0], primary_utm)
    features_df['closest_secondary'] = distance_to_roadway(location_utm['geometry'][0], secondary_utm)

    # Calculate distance to nearest intersections
    features_df['corner_dist'] = nearest_intersection(location_utm['geometry'][0], intersections_utm['geometry'])
    features_df['signal_dist'] = nearest_intersection(location_utm['geometry'][0], traffic_signals['geometry'])
    
    # 3. Create Census and city zoning based features (quick to do it with joins)
    features_df = gpd.sjoin(features_df, all_zoning, how='left', op='intersects')
    features_df = features_df.drop(['index_right'], axis = 1)
    features_df = gpd.sjoin(features_df, census_tracts, how='left', op='intersects')

    #calculate distance to industry
    features_df['industry_dist'] = distance_to_zoning(location_utm['geometry'][0], industry_utm)

    # 4. Weather features (wind and temp)
    lons = np.array(features_df['Longitude'].values)
    lats = np.array(features_df['Latitude'].values)
    
    winds = np.zeros(features_df.shape[0])
    with rasterio.open('data/wc2.0_30s_wind/avg_wind.tif') as src:
        for i, val in enumerate(src.sample(zip(lons, lats))):
            winds[i] = val

    temp = np.zeros(features_df.shape[0])
    with rasterio.open('data/wc2.0_30s_tavg/avg_ta.tif') as src:
        for i, val in enumerate(src.sample(zip(lons, lats))):
            temp[i] = val

    # remove below zero values
    winds = np.where(winds < 0, np.nan, winds)
    temp = np.where(temp < 0, np.nan, temp)

    features_df['wind'] = winds
    features_df['temp'] = temp


    ## C. Model the NO2 exposures

    # Extract used variables and create dummies
    X_vars = ['road_type', 'closest_highway', 'closest_primary', 'closest_secondary', 'corner_dist', 'signal_dist',
         'zone', 'pop_den', 'industry_dist', 'wind', 'temp']

    X = features_df[X_vars].copy()

    # Re-create dummy variable columns for input
    # for road type....
    X['road_type_motorway'] = np.where(X.road_type == 'motorway', 1, 0)
    X['road_type_primary'] = np.where(X.road_type == 'primary', 1, 0)
    X['road_type_residential'] = np.where(X.road_type == 'residential', 1, 0)
    X['road_type_secondary'] = np.where(X.road_type == 'secondary', 1, 0)
    X['road_type_tertiary'] = np.where(X.road_type == 'tertiary', 1, 0)
    X['road_type_unclassified'] = np.where(X.road_type == 'unclassified', 1, 0)

    # for zoning...
    X['zone_commercial'] = np.where(X.road_type == 'commercial', 1, 0)
    X['zone_industrial'] = np.where(X.road_type == 'industrial', 1, 0)
    X['zone_mixed'] = np.where(X.road_type == 'mixed', 1, 0)
    X['zone_open_space'] = np.where(X.road_type == 'open_space', 1, 0)
    X['zone_residential'] = np.where(X.road_type == 'residential', 1, 0)

    #drop extra columns following dummy variables
    X = X.drop(['zone', 'road_type'], axis = 1)

    ## D. Run the model!!!
    no2_model = joblib.load("models/xgb_no2.pkl")
    bc_model = joblib.load("models/xgb_bc.pkl")

    no2 = no2_model.predict(X)
    bc = bc_model.predict(X)

    X['NO2'] = no2
    X['BC'] = bc

    # Add lat long columns for later plotting
    X['Latitude'] = features_df['Latitude']
    X['Longitude'] = features_df['Longitude']
    X['address'] = address

    return X.to_json(orient='split')

@app.callback(
    Output('output-container-button', 'children'),
    [Input('intermediate_value', 'children'),
     Input('dropdown-pollutant', 'value')])

def get_estimate(model_df, pollutant):

    model_df = pd.read_json(model_df, orient='split')
    geolocation = Point(model_df['Longitude'].iloc[0], model_df['Latitude'].iloc[0])

    median_NO2 = 13
    median_BC = 0.4
    NO2_diff = ((model_df.ix[0, 'NO2'] - median_NO2)/median_NO2) * 100
    BC_diff = ((model_df.ix[0, 'BC'] - median_BC)/median_BC) * 100

    if geolocation.within(region.geometry[0]) == False:
        return "Your query is out of range. Please limit to Oakland, Berkeley, Emeryville, Albany, or El Cerrito."


    if median_NO2 < model_df.ix[0, 'NO2'] < (median_NO2 + 5):
    	alarm = 'moderate'
    else:
    	alarm = 'HIGH'

    if pollutant == 'no2':
        if NO2_diff > 0:
            return('''Health risk is **{0}** at this location. 
            	Exposures are {1}% above average.'''.format(alarm, np.round(NO2_diff, 1)))
        else:
            return('''You are at no elevated health risk. 
            	Exposures are {0}% below than the regional average.'''.format(abs(np.round(NO2_diff, 1))))
    else:
        if BC_diff > 0:
            return('''Health risk is **{0}** at this location. 
            	Exposures are {1}% above average.'''.format(alarm, np.round(BC_diff, 1)))
        else:
            return('''You are at no elevated health risk. 
            	Exposures are {0}% below than the regional average.'''.format(abs(np.round(BC_diff, 1))))


@app.callback(
	Output('suggestions', 'children'),
	[Input('intermediate_value', 'children')])

def create_suggestions(model_df):
    model_df = pd.read_json(model_df, orient='split')
    geolocation = Point(model_df['Longitude'].iloc[0], model_df['Latitude'].iloc[0])
    
    healthy_hoods = get_healthy_suggestions(geolocation, neighborhoods)

    hoods_statement = str("The following neighborhoods are healthier choices within your budget (healthiest alternative ranked first): " + 
        ", ".join(healthy_hoods.sort_values(by = 'no2')['Name'].values))
    # values_statement = str('.  These neighborhoods have ' + 
    #     ", ".join(healthy_hoods.sort_values(by = 'no2')['percent_diff'].astype(str)) + ' percent lower exposure rates.')

    hoods_statement = html.Div([
    						html.P("The following neighborhoods are healthier choices within your budget:", 
    							style = {'fontSize': 32}),
    						html.P("(healthiest alternative ranked first)",
    							style = {'fontSize': 24}),
    						html.P(", ".join(healthy_hoods.sort_values(by = 'no2')['Name'].values),
    							style = {'fontSize': 32, 'fontStyle': 'bold'})
    						])

    if healthy_hoods.shape[0] == 0:
        return 'There are no healthier neighborhoods nearby in your budget (within 1 mile).'
    else:
        return(hoods_statement)

# @app.callback(
#     Output('datatable', 'children'),
#     [Input('intermediate_value', 'children')])

# # def create_table(model_df):
# #     model_df = pd.read_json(model_df, orient='split')
# #     geolocation = Point(model_df['Longitude'].iloc[0], model_df['Latitude'].iloc[0])
    
# #     healthy_hoods = get_healthy_suggestions(geolocation, neighborhoods)
# #     return pd.DataFrame(healthy_hoods).to_dict('records')

# def generate_table(model_df, max_rows=10):

#     model_df = pd.read_json(model_df, orient='split')
#     geolocation = Point(model_df['Longitude'].iloc[0], model_df['Latitude'].iloc[0])
#     healthy_hoods = get_healthy_suggestions(geolocation, neighborhoods)

#     return html.Table(
#         # Header
#         [html.Tr([html.Th(col) for col in healthy_hoods.columns])] +

#         # Body
#         [html.Tr([
#             html.Td(healthy_hoods.iloc[i][col]) for col in healthy_hoods.columns
#         ]) for i in range(min(len(healthy_hoods), max_rows))]
#     )


@app.callback(
    Output('address_map', 'figure'),
    [Input('intermediate_value', 'children'),
     Input('dropdown-pollutant', 'value')])

def location_map(address_df, pollutant):
    address_df = pd.read_json(address_df, orient='split')

    if pollutant == 'no2':
        address_map = map_estimate(address_df, address_df['NO2'],
            lat = address_df.ix[0, 'Latitude'], lon=address_df.ix[0, 'Longitude'])
    if pollutant == 'bc':
        address_map = map_estimate(address_df, address_df['BC'],
            lat = address_df.ix[0, 'Latitude'], lon=address_df.ix[0, 'Longitude'])
    
    return address_map

if __name__ == '__main__':
    no2_model = joblib.load("models/xgb_no2.pkl")
    bc_model = joblib.load("models/xgb_bc.pkl")
    app.run_server(debug=True)