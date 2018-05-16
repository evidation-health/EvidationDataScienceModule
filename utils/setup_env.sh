#!/bin/bash

# NOTE: Be sure to clone the EvidationDataScienceModule into yoru home
# dir for this script to work

# Create python 3 environment
python3 -m evi_env ~/.

# Source virtual env
source ~/evi_env/bin/activate

# Pip install requirements: Note this assumes the EDSM in in your home directory
pip install -r ~/EvidationDataScienceModule/requirements.txt

# Add env to jupyter kernel
ipython kernel install --user --name=evi_env

echo "You are good to go!"

