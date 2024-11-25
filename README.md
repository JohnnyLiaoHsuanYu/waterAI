# WaterAI
This predictive model forecasts water restriction levels across regions over Taiwan.

Using the past decade's hydrological data, meteorological reanalysis data, meteorological downscaling data, and geographical information, machine learning models are trained to predict water conditions for the next month. This process involves data collection, cleaning, and transformation. The data is represented in a grid format with a spatial resolution of 1 km. Decision trees, random forests, and XGBoost models are explored for training, testing and optimization. The final results are presented using objective and nationally standardized water condition indicators (Water Resources Agency, MOEA).

Key variables include future dates, cumulative rainfall from the previous month, projected reservoir storage rates, and geographic coordinates. The prediction results are rounded and classified into 0, 1, 2, 3, and 4, representing normal water conditions, slightly tight water conditions, first-stage restrictions, second-stage restrictions, and third-stage restrictions, respectively.
