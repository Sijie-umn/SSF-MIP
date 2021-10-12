
## Learning and Dynamical Models for Sub-seasonal Climate Forecasting: Comparison and Collaboration

The three folders contain the code used in the He et al. *"[Learning and Dynamical Models for Sub-seasonal Climate Forecasting: Comparison and Collaboration](https://arxiv.org/abs/2110.05196)"*. 

## Groundtruth

The folder corresponds to Section 3, which contains the code to generate the ground truth dataset (step-by-step from the raw data). Cdo, Matlab, and Python are needed to run the code files. 


## SubX

The folder corresponds to Section 4. The code files are used for extracting the SubX forecasts from GMAO-GEOS and NCEP-CFSv2. The code is flexible to be extended for other SubX models.

## SSF_mip

The folder corresponds to Section 5 and Section 6. It contains the code files for data extraction, data preprocessing, and Machine Learning-based SSF model training and evaluation, as well as model comparison with SubX forecasts. The codebase can be easily used for training and evaluating more ML models, which can be used to replicate and hopefully extend our work.


There is one separate README.md file in each folder that provides more details.


