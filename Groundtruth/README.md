The code in this folder is used for constructing the ground truth dataset. The original data is downloaded from https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html. For running the code, one needs to put the raw data in the same folder and uses cdo, Matlab, and python.

## Getting started


***Step 1*** - Compute the mean for each day (daily tmp2m = 0.5 * daily max tmp2m + 0.5 * daily min tmp2m

Run mip\_tmp2m.sh as 

    chmod +x mip_tmp2m.sh
    ./mip_tmp2m.sh


***Step 2*** - Regrid the data to the resolution as 1-degree latitude by 1-degree longitude

Run Matlab code regrid\_data.m


***Step 3*** - Covert the mat files to a pandas multi-index DataFrame

Run python code regrid\_mat2h5.py


***Step 4*** - Fill in missing values in the data

Run python code filling\_missing\_values.py
(Some backup missing value filling functions are in utils.py)


***Step 5*** - Compute climatology

 - ***Step 5.1*** - Compute the long-term average for each month-day combination and each grid point
  Run python code create\_climatology\_1.py
 - ***Step 5.2*** - Smooth climatology using moving average with a window size of 31 days
  Run Matlab code smooth\_climo.m
 - ***Step 5.3*** - Extract the climatology from the mat file and save it as a pandas multi-index DataFrame

***Step 6*** - Compute the target variables (average tmp2m anomalies for weeks 3 & 4 for each grid point)

Run python code create\_target\_variables.py

The target variable is saved in the file tmp2m\_western\_us\_anom\_rmm.h5



