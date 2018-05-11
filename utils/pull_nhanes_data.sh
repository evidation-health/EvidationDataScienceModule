#!/bin/bash

# Note: This script defaults to moving the pulled data into the user's home directory

# Pull the .h5 NHANES datastore down from the public bucket on s3
wget --no-check-certificate https://s3.amazonaws.com/evi-ds-module/nhanes_data.tar.gz
mv nhanes_data.tar.gz ~/.

# Unzip the file
tar -xvzf ~/nhanes_data.tar.gz
