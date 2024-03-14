"""
Processes the SMIRK dataset.
Combines the individual frame sequences into AVI videos.
Outputs the result in the data/smirk directory.
"""

import os
import argparse


FPS = 10
SCALE = "1280:720"
OUTPUT_PATH = "./data/smirk"


def check_data_exists(video_id: str) -> bool:
    """Given a video ID, check if it already exists in the data directory"""
    video_path = os.path.join(OUTPUT_PATH, video_id, "original.avi")
    return os.path.exists(video_path)


def process_data(data_path, mode, max_videos):
    raise NotImplementedError()


def main():
    parser = argparse.ArgumentParser(description="")
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

    args = parser.parse_args()

    # If max_videos is not specified or specified as 0, process all videos
    max_videos = args.max_videos if args.max_videos > 0 else float("inf")

    process_data(args.data_path, args.mode, max_videos)


if __name__ == "__main__":
    main()
