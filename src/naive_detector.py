"""
This script attempts to find the event window in a video from a set of morton codes
"""

import os
import argparse
import pandas as pd
from typing import Union, Tuple

import helper


MAX_GAP_SEC = 2  # The max allowed gap between any two cells
MAX_LENGTH_SEC = 10  # The max allowed total event time
MIN_LENGTH_SEC = 1  # The min allowed total event time


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

    return parser.parse_args()


def frames_to_seconds(nr_frames: int, fps: int):
    return nr_frames / fps


def sequence_meets_requirements(sequence, required_cell_subsets, fps) -> bool:
    """
    Checks if a sequence of tuples, e.g. [(1, 10), (2, 20)] meets the
    requirements to be an event.
    The first element of the tuples represents the cell ID, and the second
    is the frame ID.
    """
    if len(sequence) < 2:
        return False

    total_nr_frames = abs(sequence[-1][1] - sequence[0][1])
    total_nr_seconds = frames_to_seconds(total_nr_frames, fps)

    if not MIN_LENGTH_SEC <= total_nr_seconds <= MAX_LENGTH_SEC:
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


def find_valid_sequences(tuples_list, fps):
    valid_sequences = []
    current_sequence = [tuples_list[0]]

    for i in range(1, len(tuples_list)):
        prev_tuple = tuples_list[i - 1]
        current_tuple = tuples_list[i]

        frames_diff = abs(current_tuple[1] - prev_tuple[1])
        seconds_diff = frames_to_seconds(frames_diff, fps)

        if (current_tuple[0] >= prev_tuple[0] and seconds_diff <= MAX_GAP_SEC):
            current_sequence.append(current_tuple)
        else:
            if len(current_sequence) > 1:
                valid_sequences.append(current_sequence)
            current_sequence = [current_tuple]

    if len(current_sequence) > 1:
        valid_sequences.append(current_sequence)

    return valid_sequences


def get_sequence(cell_values, required_cell_subsets, fps):
    """ """
    path = []

    for _, row in cell_values.iterrows():
        curr_frame_id = row["frame_id"]
        non_zero_cell_value_indices = [
            int(cell.split("cell")[1]) for cell in row.index[1:] if row[cell] > 0
        ]

        if not non_zero_cell_value_indices:
            continue

        for cell_index in non_zero_cell_value_indices:
            path.append((cell_index, curr_frame_id))

    if not path:
        return None

    valid_sequences = []

    for sequence in find_valid_sequences(path, fps):
        if sequence_meets_requirements(sequence, required_cell_subsets, fps):
            valid_sequences.append(sequence)

    if valid_sequences:
        return get_sequence_with_most_cells(valid_sequences)
    else:
        return None


def detect_event(
    cell_values, required_cell_subsets, fps
) -> Union[Tuple[Tuple[int, int], str], Tuple[None, None]]:
    """
    Returns the event window frame interval and the type, e.g. (30, 50, "left")
    In case there is no event, returns: None, None
    """
    sequence_left = get_sequence(cell_values, required_cell_subsets, fps)
    sequence_right = get_sequence(cell_values[::-1], required_cell_subsets, fps)

    if sequence_left:
        return (sequence_left[0][1], sequence_left[-1][1]), "left"
    elif sequence_right:
        return (sequence_right[-1][1], sequence_right[0][1]), "right"
    else:
        return None, None


def main(data_path: str, config_path: str):
    # Load config
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]
    detector_config = config["detector_config"]

    # Config variables for detector
    required_cell_subsets = detector_config["required_cell_subsets"]

    df_event_window = pd.DataFrame(
        columns=["video_id", "event_detected", "start_frame", "end_frame"]
    )

    for _, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if not os.path.isfile(os.path.join(target_path, "cell_values.csv")):
            tqdm_obj.write(
                f"Skipping {video_id}: cell_values.csv does not exist at {target_path}"
            )
            continue

        cell_values = pd.read_csv(os.path.join(target_path, "cell_values.csv"), sep=";")

        event_window, scenario_type = detect_event(cell_values, required_cell_subsets, grid_config["fps"])

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

    df_event_window.to_csv(
        os.path.join(data_path, "event_window.csv"), sep=";", index=False
    )

    helper.save_config(detector_config, data_path, "detector_config.yml")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path, args.config_path)
