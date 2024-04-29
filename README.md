# ted-sfc

_TODO_ Paper abstract here, plus any more info

## Running TED-SFC

### 1. Prerequisites

- [Conda](https://docs.anaconda.com/free/miniconda/index.html)
- [FFmpeg](https://ffmpeg.org/download.html)
- NVIDIA CUDA-enabled GPU (optional)

Note: While the pipeline does support CPU, it has only been tested with NVIDIA CUDA-enabled GPUs.

### 2. Create environment

```bash
conda create -n pyTED python=3.9
conda activate pyTED
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### 3. Prepare data

Make sure the dataset follows the structure below.

```txt
data/
├── [dataset_name]/
│   ├── 1/             # folder for video 1
│   │   ├── file_1
|   |   ├── ....
│   │   ├── file_n
│   ├── ...
│   ├── n/             # folder for video n
```

Scripts for processing datasets into the correct structure are provided in `src/scripts`. The project currently supports the following datasets:

- [SMIRK](https://www.ai.se/en/labs/data-factory/datasets/smirk-dataset)
- [ZOD](https://www.zod.zenseact.com)

```bash
# Running the script to format the ZOD dataset, for example
python src/scripts/zod/process.py path/to/original/dataset data/zod --mode random --nr-videos 10
```

**Important**: Depending on the Conda environment, ffmpeg may not work. If you cannot process the datasets, deactivate the environment.

### 4. Optional: Enable GPU-accelerated Optical Flow (NVIDIA CUDA-enabled GPUs only)

To enable running the optical flow model on the GPU, compile opencv from source with the cudaoptflow module:

1. Create a new conda environment, `pyTED-cudacv`, using the instructions from [Step 2](#2-create-environment)
1. Uninstall the current version of opencv `pip uninstall opencv-python
1. Follow [this guide](https://danielhavir.com/notes/install-opencv/) by Daniel Havir. Note:
    - Get the latest versions of opencv and opencv_contrib from the official repositories: [opencv](https://github.com/opencv/opencv/releases) and [opencv_contrib](https://github.com/opencv/opencv_contrib/tags)
    - Use the newly created environment `pyTED-cuda-cv` instead of `cv`
    - Replace references to python3.6 with python3.9
    - Ensure all the environment variables are correctly defined before running the cmake command. Example values:
        - `$python_exec: /home/elias/miniconda3/envs/pyTED-cuda-cv/bin/python`
        - `$include_dir: /home/elias/miniconda3/envs/pyTED-cuda-cv/include/python3.9`
        - `$library: /home/elias/miniconda3/envs/pyTED-cuda-cv/lib/libpython3.9.so`
        - `$default_exec: /home/elias/miniconda3/envs/pyTED-cuda-cv/bin/python3.9`
1. Test your installation with `python -c "import cv2; print('CUDA is available:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"`
    - If you get an error about GCC version 12.0.0 being required, run `conda install conda-forge::libgcc-ng==12`

Only use the `pyTED-cuda-cv` environment when running the optical flow model.

### 5. Run the processing pipeline

To run the TED-SFC pipeline, run the following command:

```bash
python src/pipeline.py -d path/to/dataset -o path/to/output -c path/to/config.yml -m [mlnet | transalnet | tasednet | optical-flow] [--cpu] [--annotations-path=path/to/annotations]
```

The pipeline will extract features from the videos using the selected method, convert cell values to Morton codes and run the event detection. The results will be placed in the output directory. Evaluation will be ran if the annotations path is provided.

### 6. Evaluate

To evaluate the event detection in terms of F1-score, sensitivity, specificity and mean IoU, use the following command:

`python src/evaluate.py path/to/event_window.csv path/to/annotations.yml`

## Datasets

| ZOD Positives | ZOD (1) Negatives | ZOD (2) Negatives | ZOD (3) Negatives | ZOD (4) Negatives |
| ------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| 000011        | 000082            | 000007            | 000143            | 000024            |
| 000046        | 000084            | 000019            | 000217            | 000168            |
| 000113        | 000098            | 000236            | 000229            | 000231            |
| 000169        | 000137            | 000238            | 000230            | 000234            |
| 000237        | 000161            | 000390            | 000232            | 000411            |
| 000292        | 000162            | 000414            | 000296            | 000464            |
| 000314        | 000306            | 000530            | 000461            | 000614            |
| 000316        | 000327            | 000583            | 000541            | 000680            |
| 000383        | 000684            | 000869            | 000603            | 000705            |
| 000389        | 000864            | 000905            | 000871            | 000877            |
| 000398        | 000865            | 000935            | 000881            | 000880            |
| 000433        | 000870            | 001012            | 000956            | 001049            |
| 000521        | 000900            | 001091            | 000977            | 001300            |
| 000653        | 000934            | 001199            | 001011            | 001307            |
| 000860        | 001326            | 001273            | 001067            | 001328            |
| 000893        | 001352            | 001294            | 001200            | 001341            |
|               | 001457            | 001245            | 001295            | 001412            |
