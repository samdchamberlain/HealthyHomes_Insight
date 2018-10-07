Front end powered by `plotly/dash` and `flask`

To run locally:

`/dash$ python app.py`

# Overview
Dash web-app that powers [healthyhomes.site](http://healthyhomes.site/). The [app.py](app.py) file contains the code for creating the web HTML layout and underlying calculations used to give address-level pollutant estimates, health risk warnings, neighborhood suggestions, and interactive heatmap. Below is an outline of the main calculations taking place:
1. Address estimates are given by using Google API to find lat, longs for a given address. From here, features required by the random forest model are re-calculated using the [feature_geometries.py](feature_geometries.py) functions in the same manner for the original GSV data.
2. Warning are given if user location is 0 to 5 ppb NO<sub>2</sub> above background (moderate risk) or over 5 ppb above background (high risk). This basic classification is based off public health studies that demonstrated 25% increased risk for chronic illness at 5ppb above background across North America.
3. Neighborhood alternatives are given for all areas within 1 mile where the rent is +/- 15% of the current neighborhood. Only areas with lower mean pollutant levels are provided.
4. The interactive map shows the current location and undryling heatmap of pollutant exposures. The dropdown allows user to toggle between black carbon and NO<sub>2</sub>

## Structure (not found on Github due to size restrictions)
*  'models' - Pickled random forest and XGBoost models - random forests used in the web-app and are too large for Github upload
* 'data' - Not added to Github due to size issues; however, important underlying data found in the main directory


