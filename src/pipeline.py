import os
import sys
import click
import datetime
import cv2
import matplotlib.pyplot as plt

from saliency.MLNet.run import main as run_saliency_mlnet
from saliency.TASEDNet.run import main as run_saliency_tasednet
from saliency.TranSalNet.run import main as run_saliency_transalnet
from grid_attention import main as run_grid_attention
from grid_optical_flow import main as run_grid_optical_flow
from detector_morton import main as run_detector_morton
from evaluate import main as run_evaluate
from morton import main as run_morton
import helper


def get_dataset_info(data_path):
    """
    Returns the number of videos and the total number of frames in the dataset
    """
    nr_videos = 0
    nr_frames = 0

    for video_dir, video_id, _ in helper.traverse_videos(data_path):
        video_path = os.path.join(video_dir, f"{video_id}.avi")

        nr_videos += 1
        nr_frames += get_total_frames(video_path)

    return nr_videos, nr_frames


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return total_frames


def check_path_exists(path, path_type):
    """
    Checks if the given path exists. Exits the program if it does not.
    """
    if not os.path.exists(path):
        print(f"{path_type} path '{path}' does not exist.")
        sys.exit(1)


def save_benchmark(output_path, start_time, end_time, nr_videos, nr_frames):
    """
    Logs the pipeline's runtime information to a text file.
    """
    duration = end_time - start_time
    sec_total = duration.total_seconds()
    sec_per_video = round(sec_total / nr_videos, 6)
    sec_per_frame = round(sec_total / nr_frames, 6)

    log_file_path = os.path.join(output_path, "pipeline_runtime_log.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Pipeline Run: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Start Time: {start_time}\n")
        log_file.write(f"End Time: {end_time}\n")
        log_file.write(f"Total Runtime: {duration}\n")
        log_file.write(f"Total nr. videos: {nr_videos}\n")
        log_file.write(f"Total nr. frames: {nr_frames}\n")
        log_file.write(f"Seconds per video: {sec_per_video}\n")
        log_file.write(f"Seconds per frame: {sec_per_frame}\n")
        log_file.write("--------------------------------------------------\n")


@click.command()
@click.option(
    "--data-path",
    "-d",
    type=click.Path(exists=True),
    prompt="Where do you have the input videos?",
    help="Path to the data directory.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(),
    prompt="Where should the output be saved?",
    help="Path where output will be saved.",
)
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True),
    prompt="Where is the configuration YML file located?",
    help="Path to the configuration YML file.",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["mlnet", "tasednet", "transalnet", "optical-flow"]),
    prompt="What method do you want to use?",
    help="The method or model to use.",
)
@click.option("--annotations-path", type=click.Path(), help="Path to annotations.")
@click.option("--cpu", is_flag=True, help="Use CPU instead of GPU.")
def main(data_path, output_path, config_path, method, annotations_path, cpu):
    generate_heatmaps = False
    if method in ["mlnet", "tasednet", "transalnet"]:
        generate_heatmaps = click.confirm("Do you want to generate heatmaps?")

    if os.path.isdir(output_path):
        if not click.confirm(
            "Output path already exists. Do you want to overwrite it?"
        ):
            click.echo("Exiting...")
            sys.exit(1)
    else:
        click.echo("Output path does not exist. Creating it.")
        os.makedirs(output_path, exist_ok=True)

    # Set matplotlib font type
    # Font type 1 (42) is high quality and suitable for PDFs
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Get the information for benchmarking
    nr_videos, nr_frames = get_dataset_info(data_path)

    start_time = datetime.datetime.now()
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if generate_heatmaps:
        print("----------------------------------------")
        print("Generating saliency heatmaps...")
        print("----------------------------------------")
        if method == "mlnet":
            run_saliency_mlnet(data_path, output_path, config_path, cpu)
        elif method == "tasednet":
            run_saliency_tasednet(data_path, output_path, config_path, cpu)
        elif method == "transalnet":
            run_saliency_transalnet(data_path, output_path, config_path, cpu)
        else:
            raise ValueError("Invalid saliency model")

    if method in ["mlnet", "tasednet", "transalnet"]:
        print("----------------------------------------")
        print("Applying attention grid...")
        print("----------------------------------------")
        run_grid_attention(data_path, output_path, config_path)

    if method == "optical-flow":
        print("----------------------------------------")
        print("Applying optical flow grid...")
        print("----------------------------------------")
        run_grid_optical_flow(data_path, output_path, config_path, use_cpu=cpu)

    print("----------------------------------------")
    print("Generating Morton codes...")
    print("----------------------------------------")
    run_morton(output_path)

    end_time = datetime.datetime.now()
    print(f"Pipeline ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Log the runtime information
    save_benchmark(output_path, start_time, end_time, nr_videos, nr_frames)

    print("----------------------------------------")
    print("Running detector...")
    print("----------------------------------------")
    if method in ["mlnet", "tasednet", "transalnet"]:
        run_detector_morton(output_path, config_path, True)
    else:
        run_detector_morton(output_path, config_path, False)

    if annotations_path:
        print("----------------------------------------")
        print("Running evaluation...")
        print("----------------------------------------")
        run_evaluate(output_path, annotations_path)

    print("========================================")
    print(f"Pipeline completed. Output saved to '{output_path}'.")
    print("========================================")


if __name__ == "__main__":
    main()
