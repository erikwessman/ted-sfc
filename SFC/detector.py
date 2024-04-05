"""
This script attempts to find the event window in a video from a set of morton codes
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Union, Tuple

import helper

def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the directory containing the output for each video",
    )
    parser.add_argument(
        "config_path",
        help="Path to the config yml file",
    )
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--attention", help="Detect for attention", action="store_true")
    group.add_argument("--optical-flow", help="Detect for optical flow", action="store_true")

    return parser.parse_args()


def get_matching_cell_key(morton_code: float, cell_morton: dict, margin: int) -> Union[int, None]:
    for cell, (min_value, max_value) in cell_morton.items():
        if min_value - margin <= morton_code <= max_value + margin:
            return cell
    return None


def valid_sequence(sequence):
    return len(sequence) > 2


def get_sequence(morton_codes, cell_ranges, required_cell_subsets, margin):
    """
    Gets a "reasonable" sequence from a list of Morton codes and a direction
    """
    sequence = []

    for _, row in morton_codes.iterrows():
        curr_frame_id = row["frame_id"]
        curr_cell_key = get_matching_cell_key(row["morton"], cell_ranges, margin)

        if not curr_cell_key:
            continue

        if not sequence:
            sequence.append((curr_cell_key, curr_frame_id))
            continue

        prev_cell_key, _ = sequence[-1]

        if prev_cell_key <= curr_cell_key:
            sequence.append((curr_cell_key, curr_frame_id))
        elif not valid_sequence(sequence):
            sequence = [(curr_cell_key, curr_frame_id)]

    contains_required_cell_subset = True
    for required_cell_subset in required_cell_subsets:
        if not any(tup[0] in required_cell_subset for tup in sequence):
            contains_required_cell_subset = False

    if valid_sequence(sequence) and contains_required_cell_subset:
        return sequence
    else:
        return None

 

def detect_event(morton_codes, cell_ranges, required_cell_subsets, margin) -> Union[Tuple[Tuple[int, int], str], Tuple[None, None]]:
    """
    Returns the event window frame interval and the type, e.g. (30, 50, "left")
    In case there is no event, returns: None, None
    """
    sequence_left = get_sequence(morton_codes, cell_ranges, required_cell_subsets, margin)
    sequence_right = get_sequence(morton_codes[::-1], cell_ranges, required_cell_subsets, margin)

    if sequence_left:
        return (sequence_left[0][1], sequence_left[-1][1]), "left"
    elif sequence_right:
        return (sequence_right[-1][1], sequence_right[0][1]), "right"
    else:
        return None, None


def main(data_path, cell_ranges, required_cell_subsets, margin, calibration_videos):
    df_event_window = pd.DataFrame(
        columns=["video_id",  "event_detected", "start_frame", "end_frame"])

    for _, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if not os.path.isfile(os.path.join(target_path, "morton_codes.csv")):
            tqdm_obj.write(
                f"Skipping {video_id}: morton_codes.csv does not exist at {target_path}")
            continue

        morton_codes = pd.read_csv(os.path.join(target_path, "morton_codes.csv"), sep=";")

        event_window, scenario_type = detect_event(morton_codes, cell_ranges, required_cell_subsets, margin)

        event_detected = event_window is not None
        start_frame, end_frame = event_window if event_window is not None else (-1, -1)
        scenario_type = scenario_type if scenario_type is not None else ""

        df_event_window = df_event_window._append({'video_id': video_id, 'event_detected': event_detected,
                                                   'start_frame': start_frame, 'end_frame': end_frame,
                                                   'scenario_type': scenario_type}, ignore_index=True)

    df_event_window.to_csv(os.path.join(
        data_path, "event_window.csv"), sep=";", index=False)
    
    helper.save_detection_plots(data_path, calibration_videos, cell_ranges)

    print("detector.py completed.")


if __name__ == "__main__":
    args = parse_arguments()
    config = helper.load_yml(args.config_path)
    detection_config = config["attention"] if args.attention else config["optical_flow"]

    required_cell_subsets = config["required_cell_subsets"]
    calibration_videos = config["detection_calibration_videos"]

    cell_ranges = detection_config["cell_ranges"]
    margin = detection_config["margin"]

    main(args.data_path, cell_ranges, required_cell_subsets, margin, calibration_videos)
