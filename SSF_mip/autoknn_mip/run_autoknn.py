# compute the spatial/temporal cosine similarity, rmse and relative r2 for each model
import sys
import os

for subx in ['GMAO', 'NCEP']:
    cmd = "python autoknn_regression.py --subx {}".format(subx)
    print(cmd)
    os.system(cmd)
