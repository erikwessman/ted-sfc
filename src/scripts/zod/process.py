"""
Processes the Zenseact Open Dataset.
Combines the individual frame sequences into AVI videos.
Outputs the result in the data/zod directory.
"""

import os
import subprocess
import argparse
import random
from tqdm import tqdm


FPS = 10
SCALE = "1280:720"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process ZOD sequences of images into videos."
    )
    parser.add_argument("data_path", type=str, help="Path to original ZOD dataset")
    parser.add_argument(
        "output_path",
        type=str,
        default="data/zod",
        help="Output path",
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


def data_exists(video_id: str, output_path: str) -> bool:
    """Given a video ID, check if it already exists in the data directory"""
    video_path = os.path.join(output_path, video_id, f"{video_id}.avi")
    return os.path.exists(video_path)


def process_data(data_path, output_path, mode, max_videos):
    os.makedirs(output_path, exist_ok=True)

    sequences_dir = os.path.join(data_path, "sequences")
    sequence_names = os.listdir(sequences_dir)

    if mode == "random":
        random.shuffle(sequence_names)
    else:
        sequence_names.sort()

    # Get the number of videos specified, 0 for all videos
    if max_videos > 0:
        max_videos = min(len(sequence_names), max_videos)
        sequence_names = sequence_names[:max_videos]

    pbar = tqdm(sequence_names, desc="Processing folders")

    for sequence_name in pbar:
        pbar.set_description(f"Processing folder {sequence_name}")

        if data_exists(sequence_name, output_path):
            continue

        sequence_dir = os.path.join(sequences_dir, sequence_name)
        camera_front_blur_dir = os.path.join(sequence_dir, "camera_front_blur")

        if os.path.isdir(camera_front_blur_dir):
            sequence_output_dir = os.path.join(output_path, sequence_name)
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
                os.path.join(sequence_output_dir, f"{sequence_name}.avi"),
            ]

            subprocess.run(
                ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            print(f"Skipped {sequence_dir}, camera front_blur directory not found.")

        pbar.set_description("Processing folders")


if __name__ == "__main__":
    args = parse_arguments()
    process_data(args.data_path, args.output_path, args.mode, args.max_videos)
