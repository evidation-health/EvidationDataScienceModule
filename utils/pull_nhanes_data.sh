#!/bin/bash

# Note: This script defaults to moving the pulled data into the user's home directory

# Pull the .h5 NHANES datastore down from the public bucket on s3
echo "Pulling down data from s3 bucket."
wget --no-check-certificate https://s3.amazonaws.com/evi-ds-module/nhanes_data.tar.gz

# Unzip the file
echo "Unzipping tarball."
tar -xvzf nhanes_data.tar.gz

# Move the files to the user's home directory
echo "Moving data to user home"
mv nhanes_data.tar.gz ~/nhanes_data.tar.gz
mv nhanes_data.h5 ~/nhanes_data.h5

echo "You are good to go!"
