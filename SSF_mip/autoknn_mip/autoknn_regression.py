# Autoreload packages that are modified

# Load relevant packages
import numpy as np
import pandas as pd
from sklearn import *
import sys
import subprocess
from datetime import datetime, timedelta
import netCDF4
import time
from functools import partial
import os
import pickle as pkl

# Load general utility functions
from experiments_util import *
# Load functionality for fitting and predicting
from fit_and_predict import *
# Load functionality for evaluation
from skill import *
# Load stepwise utility functions
from stepwise_util import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--subx', type=str, help='subx model')
args = parser.parse_args()
subx = args.subx


def shift_df(df, shift=None, date_col='start_date', groupby_cols=['lat', 'lon']):
    """Returns dataframe with all columns save for the date_col and groupby_cols
    shifted forward by a specified number of days within each group

    Args:
       df: dataframe to shift
       shift: (optional) Number of days by which ground truth measurements
          should be shifted forward; date index will be extended upon shifting;
          if shift is None or shift == 0, original df is returned, unmodified
       date_col: (optional) name of datetime column
       groupby_cols: (optional) if all groupby_cols exist, shifting performed
          separately on each group; otherwise, shifting performed globally on
          the dataframe
    """
    if shift is not None and shift != 0:
        # Get column names of all variables to be shifted
        # If any of groupby_cols+[date_col] do not exist, ignore error
        cols_to_shift = df.columns.drop(groupby_cols + [date_col], errors='ignore')
        print(cols_to_shift)
        # Function to shift data frame by shift and extend index
        shift_df = lambda grp_df: grp_df[cols_to_shift].set_index(grp_df[date_col]).shift(shift, freq="D")
        if set(groupby_cols).issubset(df.columns):
            # Shift ground truth measurements for each group
            df = df.groupby(groupby_cols).apply(shift_df).reset_index()
        else:
            # Shift ground truth measurements
            df = shift_df(df).reset_index()
        # Rename variables to reflect shift
        df.rename(columns=dict(
            zip(cols_to_shift, [col + "_shift" + str(shift) for col in cols_to_shift])),
            inplace=True)
    return df


experiment = "regression"

#
# Choose target
#
gt_id = "contest_tmp2m"  # "contest_precip" or "contest_tmp2m"
target_horizon = "34w"  # "34w" or "56w"

#
# Set variables based on target choice
#

# Identify measurement variable name
measurement_variable = get_measurement_variable(gt_id)  # 'tmp2m' or 'precip'

# column names for gt_col, clim_col and anom_col
gt_col = measurement_variable

# clim_col = measurement_variable+"_clim"
# anom_col = get_measurement_variable(gt_id)+"_anom" # 'tmp2m_anom' or 'precip_anom'

# anom_inv_std_col: column name of inverse standard deviation of anomalies for each start_date
# anom_inv_std_col = anom_col+"_inv_std"

# Name of knn columns
knn_cols = ["knn" + str(ii) for ii in xrange(1, 21)]

#
# Create list of official contest submission dates in YYYYMMDD format
#

target_dates = pd.read_hdf('data/{}_dates.h5'.format(subx))
#
# Create list of target dates corresponding to submission dates in YYYYMMDD format
#
# target_dates = submission_dates[submission_dates.dt.year == year]

# Find all unique target day-month combinations
target_day_months = pd.DataFrame({'month': target_dates.dt.month,
                                  'day': target_dates.dt.day}).drop_duplicates()


# # Choose regression parameters
#
# Record standard settings of these parameters
setting = "knn_regression"
if setting == "knn_regression":
    # Number of KNN neighbors to use in regression
    num_nbrs = 1 if gt_id.endswith('precip') else 20
    x_cols = ['knn' + str(nbr) for nbr in xrange(1, num_nbrs + 1)] + ['ones']
    # Construct fixed lag anomaly variable names
    lags = (['29', '72'] if target_horizon == '56w' else ['29', '57']) + ['365']
    x_cols = x_cols + [measurement_variable + '_shift' + lag for lag in lags]
    # Determine margin for local regression
    margin_in_days = 56 if gt_id.endswith('precip') else None
    # columns to group by when fitting regressions (a separate regression
    # is fit for each group); use ['ones'] to fit a single regression to all points
    group_by_cols = ['lat', 'lon']
    # anom_scale_col: multiply anom_col by this amount prior to prediction
    # (e.g., 'ones' or anom_inv_std_col)
    anom_scale_col = 'ones'
    # pred_anom_scale_col: multiply predicted anomalies by this amount
    # (e.g., 'ones' or anom_inv_std_col)
    pred_anom_scale_col = 'weight'
elif setting == "stepwise_no_model_selection":
    base_col = clim_col
    x_cols = default_stepwise_candidate_predictors(gt_id, target_horizon, hindcast=False) + ['knn1']
    margin_in_days = 56
    group_by_cols = ['lat', 'lon']
    anom_scale_col = anom_inv_std_col
    pred_anom_scale_col = anom_scale_col

print x_cols


# Default regression parameter values
#
# choose first year to use in training set
first_train_year = 1991
# specify regression model
fit_intercept = False
model = linear_model.LinearRegression(fit_intercept=fit_intercept)

#
# Prepare target and feature data
#
relevant_cols = set(x_cols + ['sample_weight',
                    'start_date', 'lat', 'lon', 'year', 'ones', 'target'] + group_by_cols)
# Create dataset with relevant columns only; otherwise the dataframe is too big


target = pd.read_hdf('data/tmp2m_western_us_anom_rmm.h5')
target.reset_index(level=['start_date', 'lat', 'lon'], inplace=True)
target_shift_15 = shift_df(target, shift=29, date_col='start_date', groupby_cols=['lat', 'lon'])
target_shift_43 = shift_df(target, shift=57, date_col='start_date', groupby_cols=['lat', 'lon'])
target_shift_365 = shift_df(target, shift=365, date_col='start_date', groupby_cols=['lat', 'lon'])

target.columns = ['lat', 'lon', 'start_date', 'target']
lat_lon_date_data = target.merge(target_shift_15, on=['lat', 'lon', 'start_date'])
lat_lon_date_data = lat_lon_date_data.merge(target_shift_43, on=['lat', 'lon', 'start_date'])
lat_lon_date_data = lat_lon_date_data.merge(target_shift_365, on=['lat', 'lon', 'start_date'])

# Drop rows with missing values for any relevant column

lat_lon_date_data = lat_lon_date_data.rename(columns={'target_shift29': 'tmp2m_shift29', 'target_shift57': 'tmp2m_shift57', 'target_shift365': 'tmp2m_shift365'})

anom_col = 'target'
tic()
print "Dropping rows with missing values for any relevant columns"
relevant_lat_lon_date_cols = list(set(lat_lon_date_data.columns.tolist()) & relevant_cols)
toc()
# Add supplementary columns
tic()
print "Adding supplementary columns"
lat_lon_date_data[pred_anom_scale_col] = 1.0 / lat_lon_date_data.groupby(["start_date"])[anom_col].transform('std')

# lat_lon_date_data['anom_inv_sqrt_2nd_mom'] = 1.0/np.sqrt(
#     lat_lon_date_data.groupby('start_date')[anom_col].transform('mean')**2
#     + lat_lon_date_data.groupby('start_date')[anom_col].transform('var',ddof=0))
lat_lon_date_data['ones'] = 1.0
lat_lon_date_data['zeros'] = 0.0
# To minimize the mean-squared error between predictions of the form
# (f(x_cols) + base_col - clim_col) * pred_anom_scale_col
# and a target of the form anom_col * anom_scale_col, we will
# estimate f using weighted least squares with datapoint weights
# pred_anom_scale_col^2 and effective target variable
# anom_col * anom_scale_col / pred_anom_scale_col + clim_col - base_col
lat_lon_date_data['sample_weight'] = lat_lon_date_data[pred_anom_scale_col]**2

toc()

# Load KNN data
tic()
print "Loading KNN data"
past_days = 60
days_early = 337 if target_horizon == "34w" else 323
max_nbrs = 20
knn_dir = os.path.join("knn_mip")
knn_data = pd.read_hdf(
    os.path.join(knn_dir,
                 "knn-{}-{}-days{}-early{}-maxnbrs{}.h5".format(
                     gt_id, target_horizon, past_days, days_early, max_nbrs)))
relevant_knn_cols = list(set(knn_cols) & relevant_cols)

toc()

# Restrict data to relevant columns
tic()
print "Merge datasets"
relevant_lat_lon_date_cols = list(set(lat_lon_date_data.columns.tolist()) & relevant_cols)
data = lat_lon_date_data.loc[:, relevant_lat_lon_date_cols]
relevant_knn_cols = list(set(knn_data.columns.tolist()) & relevant_cols)
data = pd.merge(data, knn_data[relevant_knn_cols],
                on=["start_date", "lat", "lon"])

del lat_lon_date_data
del knn_data
# del date_data
toc()

# Print warning if not all x columns were included
s = [x for x in x_cols if x not in data.columns.tolist()]
if s:
    print "These x columns were not found:"
    print s


num_cores = 16
# For the target (day, month) combination, fit leave-one-year regression model
# on training set subsetted to relevant margin and generate predictions for each
# held-out year
prediction_func = rolling_linear_regression_wrapper
all_preds = pd.DataFrame()

# generic_year = 2018
for target_date_obj in target_dates:
    # Get target date on generic as a datetime object
    #     target_date_obj = datetime.strptime('{}{:02d}{:02d}'.format(
    #         generic_year, day_month.month, day_month.day), "%Y%m%d")
    print target_date_obj
    # Get number of days between start date of observation period used for prediction
    # (2 weeks ahead) and start date of target period (2 or 4 weeks ahead) + 1 day do
    # to practical constraints of submission
    start_delta = get_start_delta(target_horizon)  # 29 or 43
    # Create template for held-out years: each held-out year will run from
    # last_train_date + 1 on that year (inclusive) through last_train_date
    # of the next year (inclusive)
    last_train_date = target_date_obj - timedelta(start_delta)
    if margin_in_days is not None:
        tic()
        sub_data = month_day_subset(data, target_date_obj, margin_in_days)
        toc()
    else:
        sub_data = data
#     print(sub_data.head())
    tic()
    preds = apply_parallel(sub_data.groupby(group_by_cols),
                           prediction_func, num_cores,
                           x_cols=x_cols, last_train_date=last_train_date)

    preds = preds.reset_index()
    # Only keep the predictions from the target day and month
    preds = preds[(preds.start_date.dt.year == target_date_obj.year) &
                  (preds.start_date.dt.day == target_date_obj.day) &
                  (preds.start_date.dt.month == target_date_obj.month)]
    # Concatenate predictions
    all_preds = pd.concat([all_preds, preds])
    toc()
    # ---------------
    # Evaluate only on target dates
    # ---------------
    tic()
    skills = get_col_skill(all_preds[all_preds.start_date.isin(target_dates)], "truth", "forecast", time_average=False)
    print "running mean skill = {}".format(skills.mean())
    toc()

result = all_preds.copy()
result.to_hdf('knn_mip/results_autoknn_{}.h5'.format(subx), key='result')
