The code files in this folder are adapated from [Hwang et al](https://arxiv.org/pdf/1809.07394v3.pdf) Improving Subseasonal Forecasting in the Western US with Machine Learning. 


## Environment and packages

The code is compatible with Python 2.7 and the following packages:

  - pygrib: 2.0.2
  - netCDF4: 1.2.4
  - jpeg: 9b
  - pandas: 0.20.3
  - jupyter: 1.0.0
 -  scipy: 0.19.1
 -  py-earth: 0.1.0
 -  hdf5: 1.8.18
 -  pytables: 3.4.2

## Getting started
1. Place the groundtruth file "tmp2m\_western\_us\_anom\_rmm.h5" and the date files of SubX models, e.g. "GMAO_dates.h5" and 'NCEP_dates.h5" in the folder "knn_mip".
2. Execute the Jupyter notebook knn\_step\_1-compute\_similarities.ipynb to compute the similarities between every pair of dates.
3. Execute the Jupyter notebook knn\_step\_2-get\_neighbor\_predictions.ipynb to compute the predictions of the most similar viable neighbors of each target date.
4. Execute run\_autoknn.py to generate forecasts using AutoKNN for both the forecast period of GMAO-GEOS and NCEP-CFSv2. 
