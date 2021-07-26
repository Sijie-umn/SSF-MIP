#!/bin/bash -l
#PBS -l walltime=3:00:00,nodes=1:ppn=40,mem=1024gb
#PBS -m abe
#PBS -M lixx1166@umn.edu
cd /home/srivbane/lixx1166/S2S

module load conda
module load cuda cuda-sdk

source activate s2s


python run_xgboost_detrend_daily_nao_nino.py