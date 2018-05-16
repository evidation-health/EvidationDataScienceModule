# EvidationDataScienceModule

This repo houses all the class notebooks and utilities that will be used during the UCSB spring 2018 quarter 
Health Data Science Module led by Evidation Health.

## Environment setup
To get up and running with the required dependencies and data do:
- ```git pull https://github.com/evidation-health/EvidationDataScienceModule.git```
- ```cd EvidationDataScienceModule```
- ```./utils/pull_nhanes_data.sh```
- ```./utils/setup_env.sh```

The above will pull the processed NHANES data down from a public bucket in s3, and will setup
a python virtual environment that pip installs the required dependencies. Note this has yet to be
thoroughly tested, and will only be tested on OSX and Linux.

## Module Overview
The module will be organized into four submodules that will all make use of the 
[2005 NHANES Dataset](https://wwwn.cdc.gov/nchs/nhanes/ContinuousNhanes/Default.aspx?BeginYear=2005).
The four modules are as follows:

1. ETL, data munging, data QC, storage strategies
2. Data visualization as a preparation strategy for inference and machine learning
3. Inferential analysis
4. Machine Learning for health data

## Important resources
Put resources here. . . .
