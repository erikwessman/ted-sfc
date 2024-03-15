# ted-sfc

![Example](assets/example.gif)

## Prerequisites

### Clone with submodules

**Important** Clone the repository with the DRIVE submodule

`git clone --recurse-submodules git@github.com:erikwessman/ted-sfc.git`

### Prepare data

The project currently supports the following datasets:

- [SMIRK](https://www.ai.se/en/labs/data-factory/datasets/smirk-dataset)
- [VAS-HD](https://www.ai.se/en/labs/data-factory/datasets/highway-dataset)
- [ZOD (for testing)](https://www.zod.zenseact.com)

Make sure the dataset follows the structure below.

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

Scripts for processing datasets into the correct structure are provided in `scripts/`

```bash
# Running the script to format the ZOD dataset, for example
python scripts/zod/process.py data/zod --mode random --max_videos 10
```

### Create environments

**TODO: Add cudatoolkit, etc**

```bash
# Create the environment for DRIVE
conda create -n pyRL python=3.7 -y
conda activate pyRL
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
pip install -r DRIVE/requirements.txt
```

```bash
# Create the environment for SFCs
conda create -n TED-SFC python=3.10
conda activate TED-SFC
pip install -r requirements.txt
```

### Setup DRIVE

Download saliency model [here]() and place in `DRIVE/models/saliency/mlnet_25.pth`

## Run the processing pipeline

```bash
./run_pipeline.sh path/to/dataset path/to/output path/to/dataset_config.yml path/to/event_config.yml [--no-heatmap]
```

The results will be placed in the output directory.

## Evaluate

Run the script `python SFC/evaluate.py`
