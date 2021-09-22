#!/bin/bash

for m in `seq 1979 2020`
do
    cdo ensmean tmax.${m}.nc tmin.${m}.nc tmean.${m}.nc
    # cdo -f nc -remapbil,grid.txt tmean.${m}.nc tmean.${m}_bli.n
done
