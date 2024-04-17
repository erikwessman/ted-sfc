import os
import sys
import argparse
import datetime
import cv2

from saliency.MLNet.run import main as run_saliency_mlnet
from saliency.TASEDNet.run import main as run_saliency_tasednet
from saliency.TranSalNet.run import main as run_saliency_transalnet
from grid_attention import main as run_grid_attention
from grid_optical_flow import main as run_grid_optical_flow
from morton import main as run_morton
import helper


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to run data processing pipeline."
    )

    parser.add_argument("data_path", help="Path to the data directory.")
    parser.add_argument("output_path", help="Path where output will be saved.")
    parser.add_argument("config_path", help="Path to the configuration YML file.")
    parser.add_argument(
        "--heatmap", help="Generate heatmaps.", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--saliency-model",
        choices=["mlnet", "tasednet", "transalnet"],
        default="mlnet",
        help="Generate heatmaps.",
    )
    parser.add_argument(
        "--cpu", help="Use CPU instead of GPU.", action=argparse.BooleanOptionalAction
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--attention",
        help="Use attention mechanism.",
        action=argparse.BooleanOptionalAction,
    )
    group.add_argument(
        "--optical-flow",
        help="Use optical flow.",
        action=argparse.BooleanOptionalAction,
    )

    return parser.parse_args()


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
    sec_per_video = sec_total / nr_videos
    sec_per_frame = sec_total / nr_frames

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


def main(
    data_path,
    output_path,
    config_path,
    generate_heatmaps,
    saliency_model,
    attention,
    optical_flow,
    use_cpu,
):
    check_path_exists(data_path, "Data")
    check_path_exists(config_path, "Config")

    if os.path.isdir(output_path):
        response = (
            input("Output path already exists. Do you want to overwrite it? (y/[n]): ")
            .strip()
            .lower()
        )
        if response != "y":
            print("Exiting...")
            sys.exit(1)
    else:
        print("Output path does not exist. Creating it.")
        os.makedirs(output_path, exist_ok=True)

    # Get the information for benchmarking
    nr_videos, nr_frames = get_dataset_info(data_path)

    start_time = datetime.datetime.now()
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if generate_heatmaps and attention:
        print("----------------------------------------")
        print("Generating saliency heatmaps...")
        print("----------------------------------------")
        if saliency_model == "mlnet":
            run_saliency_mlnet(data_path, output_path, config_path, use_cpu)
        elif saliency_model == "tasednet":
            run_saliency_tasednet(data_path, output_path, config_path, use_cpu)
        elif saliency_model == "transalnet":
            run_saliency_transalnet(data_path, output_path, config_path, use_cpu)
        else:
            raise ValueError("Invalid saliency model")

    if attention:
        print("----------------------------------------")
        print("Applying attention grid...")
        print("----------------------------------------")
        run_grid_attention(data_path, output_path, config_path)

    if optical_flow:
        print("----------------------------------------")
        print("Applying optical flow grid...")
        print("----------------------------------------")
        run_grid_optical_flow(data_path, output_path, config_path, use_cpu=use_cpu)

    print("----------------------------------------")
    print("Generating Morton codes...")
    print("----------------------------------------")
    run_morton(output_path)

    end_time = datetime.datetime.now()
    print(f"Pipeline ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Log the runtime information
    save_benchmark(output_path, start_time, end_time, nr_videos, nr_frames)

    print("========================================")
    print(f"Pipeline completed. Output saved to '{output_path}'.")
    print("========================================")


if __name__ == "__main__":
    args = parse_arguments()

    if args.optical_flow:
        args.heatmap = False

    main(
        args.data_path,
        args.output_path,
        args.config_path,
        args.heatmap,
        args.saliency_model,
        args.attention,
        args.optical_flow,
        args.cpu,
    )
