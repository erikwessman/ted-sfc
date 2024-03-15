# ted-sfc

## Prerequisites

### Clone with submodules

Clone the repository with the DRIVE submodule

`git clone --recurse-submodules git@github.com:erikwessman/ted-sfc.git`

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

### Create environments

**TODO: Add cudatoolkit, etc**

```bash
# Create the environment for DRIVE
conda create -n pyRL python=3.7 -y
```

```bash
# Create the environment for SFCs
conda create -n TED-SFC python=3.10
pip install -r requirements.txt
```

### Setup DRIVE

Download saliency model [here]() and place in `DRIVE/models/saliency/saliency_model.pth`

## Run the complete pipeline

```bash
./run_pipeline.sh path/to/dataset path/to/output path/to/dataset/config path/to/event/config
```
