# ted-sfc

![Example](assets/example.gif)

## Prerequisites

### Create environment

```bash
conda create -n pyTED python=3.9
conda activate pyTED
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### Prepare data

The project currently supports the following datasets:

- [SMIRK](https://www.ai.se/en/labs/data-factory/datasets/smirk-dataset)
- [ZOD](https://www.zod.zenseact.com)

Make sure the dataset follows the structure below.

```txt
data/
├── [dataset_name]/
│   ├── 1/             # folder for video 1, must include the video
│   │   ├── file_1
|   |   ├── ....
│   │   ├── file_n
│   ├── ...
│   ├── n/             # folder for video n, must include the video
```

Scripts for processing datasets into the correct structure are provided in `src/scripts`

```bash
# Running the script to format the ZOD dataset, for example
python src/scripts/zod/process.py data/zod --mode random --max_videos 10
```

## Run the processing pipeline

```bash
python src/run_pipeline.py path/to/dataset path/to/output path/to/config.yml [--attention or --optical-flow] [--heatmap]
```

The results will be placed in the output directory.

## Run detector

This script will attempt to detect events given a list of Morton codes

`python src/detector.py path/to/data path/to/config.yml [--attention or --optical-flow]`

## Evaluate

`python src/evaluate_f1.py path/to/event_window.csv path/to/ground_truth.yml`
