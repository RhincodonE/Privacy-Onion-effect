#!/bin/bash

# Base YAML configuration file path
yaml_file="./configs/config_shadow.yaml"
original_yaml_file="./configs/original_config_shadow.yaml"
number_shadow_model=16

# Make a copy of the original YAML to preserve it
cp $yaml_file $original_yaml_file

for i in $(seq 1 $number_shadow_model); do
    sed -i "s|in_model_save_path: ./data/models_shadow/.*|in_model_save_path: ./data/models_shadow/model_in_${i}.pth|" $yaml_file
    sed -i "s|out_dataset: ./data/tmp_shadow/cifar_out.*|out_dataset: ./data/tmp_shadow/cifar_out_${i}.beton|" $yaml_file
    sed -i "s|in_dataset: ./data/tmp_shadow/cifar_in.*|in_dataset: ./data/tmp_shadow/cifar_in_${i}.beton|" $yaml_file
    sed -i "s|out_dataset_index: ./data/tmp_shadow/out.*|out_dataset_index: ./data/tmp_shadow/out_${i}.npy|" $yaml_file
    sed -i "s|in_dataset_index: ./data/tmp_shadow/in.*|in_dataset_index: ./data/tmp_shadow/in_${i}.npy|" $yaml_file

    python ./scripts/write_datasets_shadow.py --config-file $yaml_file

    python ./scripts/train_shadow.py --config-file $yaml_file

done

cp $original_yaml_file $yaml_file

rm $original_yaml_file

echo "Shadow models generation done!"
