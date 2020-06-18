#!/bin/bash

PYTHON='/usr/bin/python'
CONFIG_DIR='config/*'

for config_file in $CONFIG_DIR; do
  echo "Running config file $config_file"
  $PYTHON run.py --config-file $config_file
done
