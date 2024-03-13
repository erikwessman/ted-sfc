# ted-sfc

### Prepare data

1. Put your dataset in data/
1. If dataset is not videos, convert frames to videos. Scripts for ZOD, ... provided in /scripts
1. Make sure data/ follows the following structure:

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

### Run the following command: `python main.py --dataset_name=... --model_name=... --...`

### Check the results in /output
