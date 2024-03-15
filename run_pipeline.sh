#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 data_path output_path dataset_config_path event_config_path"
    exit 1
fi

data_path=$1
output_path=$2
dataset_config_path=$3
event_config_path=$4

if [ ! -d "$data_path" ]; then
    echo "Data path does not exist."
    exit 1
fi

if [ ! -e "$dataset_config_path" ]; then
    echo "Dataset config path does not exist."
    exit 1
fi

if [ ! -e "$event_config_path" ]; then
    echo "Event config path does not exist."
    exit 1
fi

if [ ! -d "$output_path" ]; then
    echo "Output path does not exist. Attempting to create it."
    mkdir -p "$output_path"
fi

conda activate pyRL

echo "Executing main_saliency..."
python DRIVE/main_saliency.py "$data_path" "$output_path" "$dataset_config_path"

conda activate TED-SFC

echo "Executing grid_attention.py..."
python SFC/grid_attention.py "$data_path" "$output_path" "$dataset_config_path" "$event_config_path"

echo "Executing z_curve.py..."
python SFC/z_curve.py "$output_path"

echo "Pipeline done"
