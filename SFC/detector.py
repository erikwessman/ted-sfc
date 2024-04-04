"""
This script attempts to find the event window in a video from a set of morton codes
"""
import os
import argparse
import pandas as pd
from typing import Union, Tuple

import helper

CELL_MORTON = {
    1: (4.096e-07, 4.096e-07),
    2: (8.192e-07, 8.192e-07),
    3: (1.6384e-06, 1.6388e-06),
    4: (5.2e-08, 3.2768e-06),
    5: (6.5536e-06, 6.5536e-06),
    6: (1.31072e-05, 1.31104e-05),
}

MARGIN = 0
ENTER_CELLS = {1, 2, 3}
EXIT_CELLS = {4, 5, 6}


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the directory containing the output for each video",
    )
    return parser.parse_args()


def get_matching_cell_key(morton_code: float, cell_morton: dict) -> Union[int, None]:
    for cell, (min_value, max_value) in cell_morton.items():
        if min_value - MARGIN <= morton_code <= max_value + MARGIN:
            return cell
    return None


def valid_sequence(sequence):
    return len(sequence) > 2


def get_sequence(morton_codes):
    """
    Gets a "reasonable" sequence from a list of Morton codes and a direction
    """
    sequence = []

    for index, row in morton_codes.iterrows():
        curr_frame_id = row["frame_id"]
        curr_cell_key = get_matching_cell_key(row["morton"], CELL_MORTON)

        if not curr_cell_key:
            continue

        if not sequence:
            sequence.append((curr_cell_key, curr_frame_id))
            continue

        prev_cell_key, prev_frame_id = sequence[-1]

        if prev_cell_key <= curr_cell_key:
            sequence.append((curr_cell_key, curr_frame_id))
        elif not valid_sequence(sequence):
            sequence = [(curr_cell_key, curr_frame_id)]

    has_enter_cells = any(tup[0] in ENTER_CELLS for tup in sequence)
    has_exit_cells = any(tup[0] in EXIT_CELLS for tup in sequence)

    if valid_sequence(sequence) and has_enter_cells and has_exit_cells:
        return sequence
    else:
        return None


def detect_event(morton_codes) -> Union[Tuple[Tuple[int, int], str], None]:
    """
    Returns the event window frame interval and the type, e.g. (30, 50, "left")
    In case there is no event, returns: None, None
    """
    sequence_left = get_sequence(morton_codes)
    sequence_right = get_sequence(morton_codes[::-1])

    if sequence_left:
        return (sequence_left[0][1], sequence_left[-1][1]), "left"
    elif sequence_right:
        return (sequence_right[-1][1], sequence_right[0][1]), "right"
    else:
        return None, None


def main(data_path):
    df_event_window = pd.DataFrame(
        columns=["video_id",  "event_detected", "start_frame", "end_frame"])

    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if not os.path.isfile(os.path.join(target_path, "morton_codes.csv")):
            tqdm_obj.write(
                f"Skipping {video_id}: morton_codes.csv does not exist at {target_path}")
            continue

        morton_codes = pd.read_csv(os.path.join(target_path, "morton_codes.csv"), sep=";")

        event_window, scenario_type = detect_event(morton_codes)

        event_detected = event_window is not None
        start_frame, end_frame = event_window if event_window is not None else (-1, -1)
        scenario_type = scenario_type if scenario_type is not None else ""

        df_event_window = df_event_window._append({'video_id': video_id, 'event_detected': event_detected,
                                                   'start_frame': start_frame, 'end_frame': end_frame,
                                                   'scenario_type': scenario_type}, ignore_index=True)

    df_event_window.to_csv(os.path.join(
        data_path, "event_window.csv"), sep=";", index=False)

    print("detector.py completed.")


if __name__ == "__main__":
    args = parse_arguments()

    main(args.data_path)
