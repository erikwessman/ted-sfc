"""
This script attempts to find the event window in a video from a set of cell values.
"""
import os
import argparse
import pandas as pd

import helper


# Constants
CONST = 1


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the directory containing the output for each video",
    )
    return parser.parse_args()


def detect_event(cell_values):
    """
    Returns the event window frame interval, e.g. [140, 175]
    In case there is no event, returns [-1, -1]
    """
    pass


def main(data_path):
    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if os.path.isfile(os.path.join(target_path, "cell_values.csv")):
            tqdm_obj.write(f"Skipping {video_id}: Cell values CSV does not exist")
            continue

        cell_values = pd.read_csv(os.path.join(target_path, "cell_values.csv"), sep=";")

        event_window = detect_event(cell_values)

        # Save the event window somewhere

    print("detector.py completed.")


if __name__ == "__main__":
    args = parse_arguments()

    main(args.data_path)
