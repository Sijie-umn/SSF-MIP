################### Configuration for Data Loading ################################
path = 'SSF_mip/'  # need to change to the absolute path of the data files
absolute_path = 'ml_subx/'  # need to change to the absolute path of the code files
rootpath_cv = absolute_path + 'random_cv/'
forecast_rootpath = absolute_path + 'forecast/'
param_path = absolute_path + 'random_cv/cv_results_test/best_parameter/'
subx_data_path = '../SubX/'  # need to change to the absolute path of the subx data
data_path = path + 'data/'
target_data_path = '../Groundtruth/'

val_years = [2015, 2014, 2013, 2012, 2011]  # years to create validation sets

# train-test split
test_years = [2017, 2018, 2019, 2020]


################ Configuration for hyper parameter tuning  ######################
# param_grid for encoder decoder model
param_grid_en_de = {'hidden_dim': [32, 64, 128],
                    'num_layers': [1, 2, 3],
                    'learning_rate': [0.0001],
                    'threshold': [3.0, 3.5, 4.0],
                    'num_epochs': [100, 200],
                    'decoder_len': [4, 11, 18],
                    'last_layer': [True, False],
                    'seq_len': [18],
                    'linear_dim': [32, 64, 128, 256],
                    'drop_out': [0.2, 0.3],
                    'ci_dim': 8}

# param_grid for XGBoost
param_grid_xgb = {'max_depth': [3, 5, 7, 9],
                  'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'gamma': [0, 0.25, 0.5, 1.0],
                  'n_estimators': [100, 150, 200],
                  'learning_rate': [0.01, 0.05, 0.1]}

# param_grid for Lasso
param_grid_lasso = {'alpha': [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]}


num_random = 30

month_range = list(range(1, 13))
model_names = ['Lasso', 'XGBoost']

cv_metric = 'rmse'
one_day = True
num_rep = 20
