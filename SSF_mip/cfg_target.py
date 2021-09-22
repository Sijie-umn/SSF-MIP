import numpy as np

################### Configuration for Data Loading ################################
path = '/SSF_mip/data/'  # need to change to the absolute path of the data files
absolute_path = '/SSF_mip/'  # need to change to the absolute path of the code files
rootpath_cv = absolute_path + 'data/random_cv/'
forecast_rootpath = absolute_path + 'data/forecast/'
param_path = absolute_path + 'data/random_cv/cv_results_test/best_parameter/'
save_path = 'data/'
# target variables
target = 'tmp2m'  # target variable: 'tmp2m' or 'precip'
shift_days = 14  # 'days to shift for target variable 14 days means 2-week ahead prediction,
forecast_range = 14  # 'forecast range' - 14 days average or summation
operation = 'mean'  # 'compute the summation or average over the forecast range
# ("mean" for temperature, "sum" for precipitation)
save_target = True  # 'flag to indicate weather to save shifted target


train_start_date = '1990-01-01'   # Set the start date for training
train_end_date = '2017-06-30'     # Set the end date for training set
test_start_date = '2017-07-01'   # Set the end date for training'
end_date = '2020-06-30'   # Set the end date for whole dataset'

# spatial temporal covariate variables
covariates_us = ['tmp2m', 'sm', 'hgt10', 'hgt500', 'slp', 'rhum.sig995']
covariates_global = []  # spatial-temporal covariates on land.
covariates_sea = ['sst']  # spatial-temporal covariates over ocean.
pacific_atlantic = True


lat_range_global = [-20, 65]   # latitude range for covariates
lon_range_global = [120, 340]   # longitude range for covariates

lat_range_us = [25.1, 48.9]  # latitude range for covariates
lon_range_us = [235.7, 292.8]  # longitude range for covariates

lat_range_sea = [-20, 65]  # latitude range for covariates
lon_range_sea = [120, 340]  # longitude range for covariates


# spatial variable
# add_spatial = True  # flag to indicate adding spatial features: may not need this flag
spatial_set = ['elevation']  # spatial variables

# temporal variable
# add_temporal = True  # flag to indicate adding temporal features: may not need
temporal_set = ['mei', 'nao', 'mjo_phase', 'mjo_amplitude', 'nino3', 'nino4', 'nino3.4', 'nino1+2', 'ssw']   # temporal variable(s)

save_cov = True    # flag to indicate weather to save covariance


# target_lat = 37.75 # 'latitude range for target variable'
# target_lon = 237.75 #'longitude range for target variable'

################### Configuration for Dataset ################################

# preprocessing
rootpath_data = absolute_path + 'data/'
savepath_data = absolute_path + 'data/'

vars = ['sst', 'sst', 'sm', 'tmp2m', 'hgt10', 'hgt500', 'slp', 'rhum.sig995']
locations = ['atlantic', 'pacific', 'us', 'us', 'us', 'us', 'us', 'us']

num_pcs = 10

# train-validation split
data_target_file = '../Groundtruth/tmp2m_western_us_anom_rmm.h5'
data_cov_file = savepath_data + 'covariates_all_pc10.h5'
target_var = 'tmp2m'

val_years = [2017, 2016, 2015, 2014, 2013, 2012]  # years to create validation sets

val_train_range = 10  # number of years in the training set (train-val split)

val_range = 28  # number of days to include in the validation set
val_freq = '7D'  # frequency to generate validation date

# train-test split
test_years = [2017, 2018, 2019, 2020]

test_train_range = 24  # number of years in the training set (train-test split)


past_ndays = 28   # number of days to aggaregate in the past: t-n,...,t-1'

past_kyears = 2  # number of years in the past to aggaregate: t-k,...,t year'

# future_mdays = 0

################ Configuration for hyper parameter tuning  ######################
# param_grid for encoder decoder model
param_grid_en_de = {'hidden_dim': [32, 64, 128],
                    'num_layers': [1, 2, 3],
                    'learning_rate': [0.0001],
                    'threshold': [3.0, 3.5, 4.0],
                    'num_epochs': [100, 200],
                    'seq_len': [4, 11, 18],
                    'linear_dim': [32, 64, 128, 256],
                    'drop_out': [0.2, 0.3]}

# param_grid for XGBoost
param_grid_xgb = {'max_depth': [3, 5, 7, 9],
                  'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'gamma': [0, 0.25, 0.5, 1.0],
                  'n_estimators': [100, 150, 200],
                  'learning_rate': [0.01, 0.05, 0.1]}

# param_grid for Lasso
param_grid_lasso = {'alpha': [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]}


num_random = 50

month_range = list(range(1, 13))
model_names = ['Lasso', 'XGBoost', 'EncoderFNN_AllSeq']
# ['EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']
cv_metric = 'rmse'
one_day = True
num_rep = 20
