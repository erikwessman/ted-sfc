"""
Processes the SMIRK dataset.
Combines the individual frame sequences into AVI videos.
"""

import os
import cv2
import argparse
import random
from tqdm import tqdm

FPS = 10
SCALE = (752, 480)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process SMIRK dataset into videos.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to original SMIRK dataset",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to desired output",
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
    video_path = os.path.join(output_path, f"{video_id}.avi")
    return os.path.exists(video_path)


def process_data(data_path: str, output_path: str, mode: str, max_videos: int):
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

    video_dirs = tqdm(video_names, desc="Processing folders")

    for video_id in video_dirs:
        video_output_dir = os.path.join(output_path, video_id)
        os.makedirs(video_output_dir, exist_ok=True)

        if data_exists(video_id, video_output_dir):
            tqdm.write(f"Skipping {video_id}: The output video already exists")
            continue

        sequence_dir = os.path.join(data_path, video_id)
        video_output_path = os.path.join(video_output_dir, f"{video_id}.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_output_path, fourcc, FPS, SCALE)

        if not os.path.isdir(sequence_dir):
            tqdm.write(f"Skipping file {sequence_dir} as it's not a directory.")
            continue

        image_files = [f for f in sorted(os.listdir(sequence_dir)) if f.endswith('.png') and not f.endswith('.labels.png')]

        for image_file in image_files:
            image_path = os.path.join(sequence_dir, image_file)
            frame = cv2.imread(image_path)
            frame = cv2.resize(frame, SCALE, interpolation=cv2.INTER_AREA)
            out.write(frame)

        out.release()


def main():
    args = parse_arguments()

    process_data(args.data_path, args.output_path, args.mode, args.max_videos)


if __name__ == "__main__":
    main()
