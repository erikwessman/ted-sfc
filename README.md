# ted-sfc

## Prerquisites

### Prepare data

1. Make sure the dataset follows the structure below.
1. Scripts for processing datasets into the correct structure are provided in `/scripts`

```txt
data/
├── [dataset_name]/
│   ├── info.json      # defines the subsets to use for calibration and testing
│   ├── 1/             # folder for video 1, must include the video
│   │   ├── file_1
|   |   ├── ....
│   │   ├── file_n
│   ├── ...
│   ├── n/             # folder for video n, must include the video
```

### Create conda environment and install requirements

// TODO: Add cudatoolkit, etc

```bash
conda create -n DRIVE_SFC python=3.7 -y
conda activate DRIVE_SFC

pip install -r requirements.txt
pip install -r SFC/requirements.txt
pip install -r DRIVE/requirements.txt
```

### Setup DRIVE

Download saliency model [here]() and place in `DRIVE/models/saliency/saliency_model.pth`

## Run the complete pipeline

```bash
./run_pipeline.sh path/to/dataset/config path/to/event/config
```
