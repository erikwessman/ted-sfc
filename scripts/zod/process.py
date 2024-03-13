"""
Processes the Zenseact Open Dataset.
Combines the individual frame sequences into AVI videos.
Outputs the result in the data/zod directory.
"""

import os
import subprocess
import argparse
import random


FPS = 10
SCALE = "1280:720"
OUTPUT_PATH = "./data/zod"


def check_data_exists(video_id: str) -> bool:
    """Given a video ID, check if it already exists in the data directory"""
    video_path = os.path.join(OUTPUT_PATH, video_id, "original.avi")
    return os.path.exists(video_path)


def process_data(data_path, mode, max_videos):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    sequences_dir = os.path.join(data_path, "sequences")
    sequence_names = os.listdir(sequences_dir)

    if mode == "random":
        random.shuffle(sequence_names)
    else:
        sequence_names.sort()

    processed_videos = 0
    for sequence_name in sequence_names:
        if processed_videos >= max_videos:
            break

        if check_data_exists(sequence_name):
            processed_videos += 1
            print(f"{sequence_name} exists, skipping...")
            continue

        sequence_dir = os.path.join(sequences_dir, sequence_name)
        camera_front_blur_dir = os.path.join(sequence_dir, "camera_front_blur")

        if os.path.isdir(camera_front_blur_dir):
            sequence_output_dir = os.path.join(OUTPUT_PATH, sequence_name)
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
                os.path.join(sequence_output_dir, "original.avi"),
            ]

            subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            print(f"Processed sequence {sequence_name}")
            processed_videos += 1
        else:
            print(f"Skipped {sequence_dir}, camera_front_blur directory not found.")


def main():
    parser = argparse.ArgumentParser(
        description="Process ZOD sequences of images into videos."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to original ZOD dataset",
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

    args = parser.parse_args()

    # If max_videos is not specified or specified as 0, process all videos
    max_videos = args.max_videos if args.max_videos > 0 else float("inf")

    process_data(args.data_path, args.mode, max_videos)


if __name__ == "__main__":
    main()
