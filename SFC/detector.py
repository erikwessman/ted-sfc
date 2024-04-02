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
ALLOWED_GAP = 30
ENTER_CELLS = set([1, 2, 3])
EXIT_CELLS = set([4, 5, 6])


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

def detect_event(morton_codes) -> Union[Tuple[int, int], None]:
    """
    Returns the event window frame interval, e.g. [140, 175]
    In case there is no event, returns None
    """

    event_start = None
    event_end = None
    cells_matched = set()
    last_matching_frame = None
    temp_cell_morton = CELL_MORTON.copy()

    for index, row in morton_codes.iterrows():
        matching_cell_key = get_matching_cell_key(row["morton"], temp_cell_morton)

        if matching_cell_key is not None:
            cells_matched.add(matching_cell_key)

            # delete cells before the matching cell, since we are looking in a particular direction
            # NOT WORKING since we may, for example, get a match with a noisey 4 before a correct match with cell 1
            # for i in range(matching_cell_key):
            #     if i in temp_cell_morton:
            #         del temp_cell_morton[i]

            if not event_start:
                event_start = row["frame_id"]
            elif row["frame_id"] - last_matching_frame <= ALLOWED_GAP:
                event_end = row["frame_id"]

            last_matching_frame = row["frame_id"]

    has_enter = bool(ENTER_CELLS & cells_matched)
    has_exit = bool(EXIT_CELLS & cells_matched)
    if event_start and event_end and has_enter and has_exit:
        return (event_start, event_end)
    else:
        return None

def main(data_path):
    event_window_map = {}
    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if not os.path.isfile(os.path.join(target_path, "morton_codes.csv")):
            tqdm_obj.write(f"Skipping {video_id}: morton_codes.csv does not exist at {target_path}")
            continue

        morton_codes = pd.read_csv(os.path.join(target_path, "morton_codes.csv"), sep=";")

        event_window = detect_event(morton_codes)

        event_window_map[video_id] = event_window

        print(f"Event window for {video_id}: {event_window}")

    # event_window_map.to_csv(os.path.join(data_path, "event_window.csv"), sep=";", index=False)
    print("detector.py completed.")


if __name__ == "__main__":
    args = parse_arguments()

    main(args.data_path)
