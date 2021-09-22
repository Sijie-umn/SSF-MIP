import pandas as pd
from datetime import datetime, timedelta
import numpy as np

idx = pd.IndexSlice


# some back up missing value filling functions
def find_gap(df, date_index, example_date):
    one_day_sample = df.loc[idx[:, :, example_date], :]
    id_example = one_day_sample.reset_index().set_index(['lat', 'lon']).index.unique()
    id_record_all = []
    for date in date_index:
        one_day = df.loc[idx[:, :, date], :]
        # temp=df.loc[idx[:,:,date],:]
        id_temp = one_day.reset_index().set_index(['lat', 'lon']).index.unique()
        id_record = []
        for id in id_example:
            if id not in id_temp:
                id_record.append(id)
        id_record_all.append(id_record.copy())
    return id_record_all


# function for filling the missing values
def fill_gap(df, date_index, id_record):
    i = -1
    for date in date_index:
        i += 1
        for id in id_record[i]:
            average_values = compute_filling_element(df, date, id)
            # print(date,id,average_values)
            df.loc[idx[id[0], id[1], date], :] = average_values
            df.sort_index(ascending=True, inplace=True)
    return df


# function for computing missing values
def compute_filling_element(df, date, id):
    idx = pd.IndexSlice
    neighbor = []
    resolution = 1.0  # 0.5 for raw data
    neighbor.append([id[0] + resolution, id[1] + resolution, date])
    neighbor.append([id[0] + resolution, id[1], date])
    neighbor.append([id[0] + resolution, id[1] - resolution, date])
    neighbor.append([id[0], id[1] + resolution, date])
    neighbor.append([id[0], id[1] - resolution, date])
    neighbor.append([id[0] - resolution, id[1] + resolution, date])
    neighbor.append([id[0] - resolution, id[1], date])
    neighbor.append([id[0] - resolution, id[1] - resolution, date])
    neighbor_saptial = pd.DataFrame(neighbor, columns=['lat', 'lon', 'start_date'])
    neighbor_saptial['start_date'] = pd.to_datetime(neighbor_saptial['start_date'])
    df_temp = df.reset_index()
    neighbor_saptial = neighbor_saptial.merge(df_temp, on=['lat', 'lon', 'start_date'])
    neighbor_saptial = neighbor_saptial.dropna()
    temp = neighbor_saptial.mean(axis=0).values
    return temp[-1:]
