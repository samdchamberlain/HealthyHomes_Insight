# HealthyHomes
Predicting traffic-related pollutant exposures in the East Bay, CA. This map-based web app predicts exposures to NO<sub>2</sub> and black carbon at individual addresses, while also providing health risk assessments and alternative, healthier neighborhoods within ones price range. [The web-app can be found here](http://healthyhomes.site/)

## Overview
This project leverages a dataset of hyperlocal air pollution mapping collected by Google Street View (GSV) and the Environmental Defense Fund ([link])(https://www.edf.org/airqualitymaps) to generate address-level estimates of pollutant exposures at homes throughout the East Bay. Currently, the model and web-app is designed to estimate exposures at addresses in Oakland, Emeryville, Berkeley, Albany, and El Cerrito, California. The basic workflow that generates the [web-app](http://healthyhomes.site/) are as follows:
* Features are generated for the location-based GSV data, as the original dataset is simply location information and gas concentrations. Additional data sources used for feature engineering include:
	1. [OpenStreetMaps](https://www.openstreetmap.org/)
	2. [US Census](https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml)
	3. [City zoning](http://opendata.mtc.ca.gov/)
	4. [Weather rasters (1 km)](http://worldclim.org/version2)
* Machine learning models are trained on the GSV data w/ features (20 total), and random forests were applied in production on the web-app.
* The underlying heatmap is generated for a point grid across all cities in the East Bay following the same procedure of feature engineering as above.
* All current rental data for Oakland is scraped from Zillow to find neighborhoods within your price range with healthier air quality.
* The web-app is created using Dash. When an address is entered, features are generated for the location on the backend and an air quality prediction is made. 

## Structure
This repository contains all of the underlying data, notebooks used for data processing, feature engineering, and modelings, as well as code used to create the web-app. The overall structure is as follows:
1. Root directory - contains the main notebooks used to engineer features, run machine learning models, generate underlying heatmap found in the web-app:
	* [Pollutant_model.ipynb](Pollutant_model.ipynb) - All data cleaning, feature engineering, and modeling for the GSV dataset. Exports final model used in the web-app.
	* [Heatmap_generation.ipynb](Heatmap_generation.ipynb) - Generates the underlying heatmap on the web-app based on feature engineering and modeling pollutant exposures for a regular grid of points across the East Bay.
	* [Zillow_scraping.ipynb](Zillow_scraping.ipynb) - Scrapes all current rentals in Oakland for the recommendations of healthier neighborhoods in a similar price range in Oakland.
	* [feature_geometries.py](feature_geometries.py) - All functions used for feature engineerings, includes distance based metrics (ex. distance from highway) and finding nearest roadtypes for example.
2. [imports](https://github.com/samdchamberlain/HealthyHomes_Insight/tree/master/Import%20notebooks) - Jupyter notebooks used for importing census data from API and aggregating global temperature and wind raster files. (FIX DEAD LINK)
3.  [dash-app](https://github.com/samdchamberlain/HealthyHomes_Insight/tree/master/dash-app) - All scripts and data required for creating the web-app. More description of the web-app can be found within its README

The full data is not uploaded to Github due to file sizes that exceed those allowed by GitHub; however, the original GSV data and heatmap output can be found within the [data] folder.
