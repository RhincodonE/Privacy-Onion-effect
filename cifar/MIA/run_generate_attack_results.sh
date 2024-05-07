#!/bin/bash

# Base YAML configuration file path
yaml_file="./configs/config_attack.yaml"
original_yaml_file="./configs/original_config_attack.yaml"

# Make a copy of the original YAML to preserve it
cp $yaml_file $original_yaml_file

python ./scripts/attack_target.py --config-file $yaml_file
python ./scripts/privacy_score.py

cp $original_yaml_file $yaml_file

rm $original_yaml_file
echo "Attacking finished!"
