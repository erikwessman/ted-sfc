"""
This script attempts to find the event window in a video from a set of morton codes
"""
import os
import argparse
import pandas as pd
from typing import Union, Tuple

import helper


MAX_GAP_S = 20
MAX_LENGTH_S = 100
MIN_LENGTH_S = 20


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_path", help="Path to the directory containing the output for each video",)
    parser.add_argument("config_path", help="Path to the config yml file",)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--attention", help="Detect for attention", action="store_true")
    group.add_argument("--optical-flow", help="Detect for optical flow", action="store_true")

    return parser.parse_args()


def get_matching_cell_key(morton_code: float, cell_morton: dict, margin: int) -> Union[int, None]:
    for cell, (min_value, max_value) in cell_morton.items():
        if min_value - margin <= morton_code <= max_value + margin:
            return cell
    return None


def sequence_meets_requirements(sequence, required_cell_subsets) -> bool:
    """
    Checks if a sequence of tuples, e.g. [(1, 10), (2, 20)] meets the
    requirements to be an event.
    The first element of the tuples represents the cell ID, and the second
    is the frame ID.
    """
    if len(sequence) < 2:
        return False

    start = sequence[0][1]
    end = sequence[-1][1]

    if end - start > MAX_LENGTH_S or end - start < MIN_LENGTH_S:
        return False

    contains_required_cell_subset = True
    for required_cell_subset in required_cell_subsets:
        if not any(tup[0] in required_cell_subset for tup in sequence):
            contains_required_cell_subset = False

    return contains_required_cell_subset


def get_sequence_with_most_cells(sequences):
    max_unique_count = -1
    list_with_max_unique = []

    for sequence in sequences:
        unique_first_elements = len(set(x[0] for x in sequence))

        if unique_first_elements > max_unique_count:
            max_unique_count = unique_first_elements
            list_with_max_unique = sequence

    return list_with_max_unique


def find_valid_sequences(tuples_list):
    valid_sequences = []
    current_sequence = [tuples_list[0]]

    for i in range(1, len(tuples_list)):
        prev_tuple = tuples_list[i - 1]
        current_tuple = tuples_list[i]

        # TODO stuff with frames and seconds
        if current_tuple[0] >= prev_tuple[0] and abs(current_tuple[1] - prev_tuple[1]) <= MAX_GAP_S:
            current_sequence.append(current_tuple)
        else:
            if len(current_sequence) > 1:
                valid_sequences.append(current_sequence)
            current_sequence = [current_tuple]

    if len(current_sequence) > 1:
        valid_sequences.append(current_sequence)

    return valid_sequences


def get_sequence(morton_codes, cell_ranges, required_cell_subsets, margin):
    """ """
    path = []

    for _, row in morton_codes.iterrows():
        curr_frame_id = row["frame_id"]
        curr_cell_key = get_matching_cell_key(row["morton"], cell_ranges, margin)

        if not curr_cell_key:
            continue

        path.append((curr_cell_key, curr_frame_id))

    if not path:
        return None

    valid_sequences = []

    for sequence in find_valid_sequences(path):
        if sequence_meets_requirements(sequence, required_cell_subsets):
            valid_sequences.append(sequence)

    if valid_sequences:
        return get_sequence_with_most_cells(valid_sequences)
    else:
        return None


def detect_event(
    morton_codes, cell_ranges, required_cell_subsets, margin
) -> Union[Tuple[Tuple[int, int], str], Tuple[None, None]]:
    """
    Returns the event window frame interval and the type, e.g. (30, 50, "left")
    In case there is no event, returns: None, None
    """
    sequence_left = get_sequence(
        morton_codes, cell_ranges, required_cell_subsets, margin
    )
    sequence_right = get_sequence(
        morton_codes[::-1], cell_ranges, required_cell_subsets, margin
    )

    if sequence_left:
        return (sequence_left[0][1], sequence_left[-1][1]), "left"
    elif sequence_right:
        return (sequence_right[-1][1], sequence_right[0][1]), "right"
    else:
        return None, None


def main(data_path: str, config_path: str, use_attention: bool):
    # Load config
    config = helper.load_yml(config_path)
    detector_config = config["detector_config"]

    # Config variables for detector
    required_cell_subsets = detector_config["required_cell_subsets"]
    calibration_videos = detector_config["detection_calibration_videos"]

    # Config variables specific to the type, either attention or OF
    type_config = (detector_config["attention"] if use_attention else detector_config["optical_flow"])
    cell_ranges = type_config["cell_ranges"]
    margin = type_config["margin"]

    df_event_window = pd.DataFrame(columns=["video_id", "event_detected", "start_frame", "end_frame"])

    for _, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if not os.path.isfile(os.path.join(target_path, "morton_codes.csv")):
            tqdm_obj.write(f"Skipping {video_id}: morton_codes.csv does not exist at {target_path}")
            continue

        morton_codes = pd.read_csv(os.path.join(target_path, "morton_codes.csv"), sep=";")

        event_window, scenario_type = detect_event(morton_codes, cell_ranges, required_cell_subsets, margin)

        event_detected = event_window is not None
        start_frame, end_frame = event_window if event_window is not None else (-1, -1)
        scenario_type = scenario_type if scenario_type is not None else ""

        df_event_window = df_event_window._append(
            {
                "video_id": video_id,
                "event_detected": event_detected,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "scenario_type": scenario_type,
            },
            ignore_index=True,
        )

    df_event_window.to_csv(os.path.join(data_path, "event_window.csv"), sep=";", index=False)

    helper.save_detection_plots(data_path, calibration_videos, cell_ranges)
    helper.save_config(detector_config, data_path, "detector_config.yml")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path, args.config_path, args.attention)
