## Learning and Dynamical Models for Sub-seasonal Climate Forecasting: Comparison and Collaboration

Code for data extraction, preprocessing, and ML-based SSF model training and evaluation, as well as model comparison with SubX forecasts.  Due to the size limitation for supplementary materials, only a small subset of the data is included as examples.

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
- pytables: 3.5.2

## Getting started
1. Clone the repo
2. Create virtual environments and install the necessary python packages listed above
3. Load raw data and ground truth data, and save it to the folder named "data" 
4. Revise configure files (cfg\_target.py and cfg\_target\_subx.py ) to adapt to the required settings

### For Machine Learning algorithms: 
1. Data loading and preprocessing:
   1. Execute load_data.py to load the subset of data needed in generating forecasts
   2. Execute run_preprocess.py to preprocess covariates and target variables separately
   3. Execute create\_covariates\_pca.py to concatenate data
   4. Execute create\_datasets.py to create training-validation sets and training-test sets
2. Hyperparameter tuning: execute run\_random\_search.py to find the best parameter by random search
3. Generate forecasts: execute main_experiments.py to train all the models and generate forecasts on test sets

### For including SubX forecasts into Machine Learning models: 
1. Data loading and preprocessing:
   1. Preprocess SubX dataset following the instructions (see the folder SubX)
   2. Execute dataset\_creation\_with\_subx.py to create training-validation sets and training-test sets
2. Hyperparameter tuning: execute run\_random\_search\_subx.py to find the best parameter by random search
3. Generate forecasts: execute main_experiments\_subx.py to train the ML models with and without including SubX forecasts in their feature sets


### For AutoKNN:
The code is adapted from [Hwang et al. 2019]. All the code files are saved in the folder "autoknn_mip". Python 2.7 is needed to execute the code files. The details of the environment and packages can be found in the GitHub repo for [Hwang et al. 2019].
 
1. Execute the Jupyter notebook knn\_step\_1-compute\_similarities.ipynb to compute the similarities between every pair of dates.
2. Execute the Jupyter notebook knn\_step\_2-get\_neighbor\_predictions.ipynb to compute the predictions of the most similar viable neighbors of each target date.
3. Execute run\_autoknn.py to generate forecasts using AutoKNN for both the forecast period of GMAO-GEOS and NCEP-CFSv2. 

### Evaluation:

Execute run_evaluation.py to evaluate the forecasting performance on all models on the forecast period of both GMAO-GEOS and NCEP-CFSv2.
