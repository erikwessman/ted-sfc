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

**Important**: Depending on the Conda environment, ffmpeg may not work. If you cannot process the datasets, deactivate the environment.

### Optional: Enable GPU-accelerated Optical Flow (NVIDIA CUDA-enabled GPUs only)

To enable running the optical flow model on the GPU, compile opencv from source with the cudaoptflow module:

- Uninstall the existing opencv package: `pip uninstall opencv-python`
- Follow [this guide](https://danielhavir.com/notes/install-opencv/) by Daniel Havir. Note:
  - Get the latest versions of opencv and opencv_contrib from the official repositories: [opencv](https://github.com/opencv/opencv/releases) and [opencv_contrib](https://github.com/opencv/opencv_contrib/tags)
  - Skip the creation of a new conda environment and use pyTED instead
  - Use python3.9 instead of python3.6
  - Ensure all the environment variables are correctly defined before running the cmake command. Example values:
    - `$python_exec: /home/elias/miniconda3/envs/pyTED/bin/python`
    - `$include_dir: /home/elias/miniconda3/envs/pyTED/include/python3.9`
    - `$library: /home/elias/miniconda3/envs/pyTED/lib/libpython3.9.so`
    - `$default_exec: /home/elias/miniconda3/envs/pyTED/bin/python3.9`
- Test your installatio with `python -c "import cv2; print('CUDA is available:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"`
  - If you get an error about GCC version 12.0.0 being required, run `conda install conda-forge::libgcc-ng==12`

## Run the processing pipeline

```bash
python src/run_pipeline.py path/to/dataset path/to/output path/to/config.yml [--attention or --optical-flow] [--heatmap] [--cpu]
```

The results will be placed in the output directory.

## Run detector

This script will attempt to detect events given a list of Morton codes

`python src/detector.py path/to/data path/to/config.yml [--attention or --optical-flow]`

## Evaluate

`python src/evaluate_f1.py path/to/event_window.csv path/to/ground_truth.yml`
