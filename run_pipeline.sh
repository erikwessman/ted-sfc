#!/bin/bash

CONDA_BIN=$(which conda)
if [ -z "$CONDA_BIN" ]; then
    echo "Conda command not found. Please make sure conda is installed and available in your PATH."
    exit 1
fi

CONDA_BASE=$(dirname $(dirname $CONDA_BIN))
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 data_path output_path dataset_config_path event_config_path [--no-heatmap]"
    exit 1
fi

data_path=$1
output_path=$2
dataset_config_path=$3
event_config_path=$4
skip_heatmap=0

for arg in "$@"
do
    if [ "$arg" == "--no-heatmap" ]; then
        skip_heatmap=1
    fi
done

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

if [ -d "$output_path" ]; then
    while true; do
        echo -n "Output path already exists. Do you want to overwrite it? (y/[n]): "
        read -r response
        if [ "$response" == "y" ]; then
            # If response is yes, break the loop and continue with the script
            break
        elif [ "$response" == "n" ] || [ -z "$response" ]; then
            # If response is no or empty, exit the script
            echo "Exiting..."
            exit 1
        else
            # If response is anything else, inform the user and repeat the question
            echo "Invalid response. Please enter 'y' for yes or 'n' for no."
        fi
    done
else
    echo "Output path does not exist. Attempting to create it."
    mkdir -p "$output_path"
fi

if [ "$skip_heatmap" -eq 0 ]; then
    conda activate pyRL

    echo "Executing main_saliency..."
    python DRIVE/main_saliency.py "$data_path" "$output_path" "$dataset_config_path"
fi

conda activate TED-SFC

echo "Executing grid_attention.py..."
python SFC/grid_attention.py "$data_path" "$output_path" "$dataset_config_path" "$event_config_path"

echo "Executing z_curve.py..."
python SFC/z_curve.py "$output_path"

echo "Pipeline done"
