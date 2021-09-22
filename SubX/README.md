The code in this folder is used for obtaining the SubX forecasts. The original data (for hindcast, climatology, and forecast of GMAO-GEOS and NCEP-CFSv2) needs to be downloaded from [SubX project](http://cola.gmu.edu/subx/data/descr.html). The data file (western_us.h5') for the spatial coverage over the western US needs to be saved in the folder as well.

## Requirements
The code is compatible with Python 3.6 and the following packages:

- numpy: 1.19.0
- pandas: 0.24.2
- joblib: 0.15.1
- pickle: 4.0
- scipy: 1.5.0
- pytorch: 1.2.0
- pytables: 3.5.2

## Getting started

***Step 0*** - Download/Compute the ensemble mean from four ensemble members for each forecast


***Step 1*** - Detect the dummy files in the SubX forecast period

Run python code check\_nan.py

***Step 2*** - Compute climatology

Run preprocess\_subx\_climo.py

***Step 3*** - Compute the average tmp2m over weeks 3 & 4 for western us for the hindcast period

Run python code preprocess\_subx\_hindcast.py

***Step 4*** - Compute the average tmp2m over weeks 3 & 4 for western us for the forecast period

Run python code preprocess\_subx\_forecast.py

***Step 5*** - Compute the target variable (the average tmp2m anomalies over weeks 3 & 4 for western us) for both hindcast and forecast period

Run python code compute\_subx_anom.py

