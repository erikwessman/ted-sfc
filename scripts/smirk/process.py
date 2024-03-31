"""
Processes the SMIRK dataset.
Combines the individual frame sequences into AVI videos.
Outputs the result in the data/smirk directory.
"""

import os
import subprocess
import argparse
import random
from tqdm import tqdm

FPS = 10
SCALE = "752:480"
OUTPUT_PATH = "./data/smirk"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process SMIRK dataset into videos.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to original SMIRK dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "random"],
        default="sequential",
        help="Order in which to process the sequences. Default is 'sequential'.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=0,
        help="Maximum number of videos to process. Default is 0 (process all).",
    )
    return parser.parse_args()


def data_exists(video_id: str) -> bool:
    """Given a video ID, check if it already exists in the data directory"""
    video_path = os.path.join(OUTPUT_PATH, video_id, "original.avi")
    return os.path.exists(video_path)


def process_data(data_path, output_path, mode, max_videos):
    os.makedirs(output_path, exist_ok=True)

    video_names = os.listdir(data_path)

    if mode == "random":
        random.shuffle(video_names)
    else:
        video_names.sort()

    # Get the number of videos specified, 0 for all videos
    if max_videos > 0:
        max_videos = min(len(video_names), max_videos)
        video_names = video_names[:max_videos]

    pbar = tqdm(video_names, desc="Processing folders")

    for video_name in pbar:
        pbar.set_description(f"Processing folder {video_name}")

        if data_exists(video_name, output_path):
            continue

        sequence_dir = os.path.join(data_path, video_name)
        camera_front_blur_dir = os.path.join(sequence_dir, "camera_front_blur")

        if os.path.isdir(camera_front_blur_dir):
            sequence_output_dir = os.path.join(output_path, video_name)
            os.makedirs(sequence_output_dir, exist_ok=True)

            ffmpeg_command = [
                "ffmpeg",
                "-framerate",
                str(FPS),
                "-pattern_type",
                "glob",
                "-i",
                os.path.join(camera_front_blur_dir, "*.jpg"),
                "-vf",
                f"scale={SCALE}",
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "veryfast",
                os.path.join(sequence_output_dir, f"{video_name}.avi"),
            ]

            subprocess.run(
                ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            print(f"Skipped {sequence_dir}, camera front_blur directory not found.")

        pbar.set_description("Processing folders")


def main():
    args = parse_arguments()

    process_data(args.data_path, args.mode, args.max_videos)


if __name__ == "__main__":
    main()
