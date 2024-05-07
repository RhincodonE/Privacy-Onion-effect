#!/bin/bash

# Base YAML configuration file path
yaml_file="./configs/config.yaml"
original_yaml_file="./configs/config_original.yaml"
number_experiment_models=128
# Make a copy of the original YAML to preserve it
cp $yaml_file $original_yaml_file

for i in $(seq 1 $number_experiment_models); do
    sed -i "s|in_model_save_path: ./models/.*|in_model_save_path: ./models/model_in_${i}.pth|" $yaml_file
    sed -i "s|out_dataset: ./tmp/cifar_out.*|out_dataset: ./tmp/cifar_out_${i}.beton|" $yaml_file
    sed -i "s|in_dataset: ./tmp/cifar_in.*|in_dataset: ./tmp/cifar_in_${i}.beton|" $yaml_file
    sed -i "s|out_dataset_index: ./tmp/out.*|out_dataset_index: ./tmp/out_${i}.npy|" $yaml_file
    sed -i "s|in_dataset_index: ./tmp/in.*|in_dataset_index: ./tmp/in_${i}.npy|" $yaml_file

    python ./scripts/write_datasets.py --config-file $yaml_file

    python ./scripts/train_cifar.py --config-file $yaml_file

done

# Cleanup: Restore original configuration file and remove backup
cp $original_yaml_file $yaml_file
rm $original_yaml_file

echo "All models trained."
