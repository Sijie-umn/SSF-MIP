# Learning and Dynamical Models for Sub-seasonal Climate Forecasting: Comparison and Collaboration (SSF-MIP)

Code for generating ground truth dataset, extract data from SubX dataset, and training and evaluating Machine Learning-based forecasting models

## Requirements
The code is compatible with Python 3.6 and the following packages:
- numpy: 1.19.0
- pandas: 0.24.2
- joblib: 0.15.1
- pickle: 4.0
- scipy: 1.5.0
- pytorch: 1.2.0
- sklearn: 0.23.1
- xgboost: 1.0.2
- xarray: 0.17.0
- netCDF4: 1.5.3

## Getting started
1. Clone the repo
2. Create virtual environments and install the necessary python packages listed above
3. Load raw data from the google drive folder and save it to a folder named "data"


## Project Structure

### SubX forecasts

### Grountruth dataset

### ML models

#### Scripts for generating forecasts
- cfg_target.py: configure file with all parameters from users
- load_data: script for a subset of data required by configure file (cfg_target.py)
- run_preprocess.py: script for data preprocessing
- create_covariates_pca.py: script for concatenating PCs from all climate variables (covariates)
- create_datasets.py: script for creating training-validation sets and training-test sets
- run_random_search.py: script for hyperparameter tuning via random search
- main_experiments: script for running experiments for all models on training and test sets
- run_evaluation.py: script for evaluate the performance of forecasting models on training and test sets


#### Functions
- data_load: folder contains functions for data loading
- preprocess: folder contains functions for data preprocessing
- hyperparameter_tuning: folder contains functions for random search
- forecasting: folder contains scripts for training forecasting models and evaluating on test sets
- evaluation: folder contains functions for evaluation
- model: collection of models implemented in experiments
- utils:utility functions
