"""
This script attempts to find the event window in a video from a set of morton codes
"""
import os
import argparse
import pandas as pd

import helper

CELL_MORTON = {
        "1": (4.096e-07, 4.096e-07),
        "2": (8.192e-07, 8.192e-07),
        "3": (1.6384e-06, 1.6388e-06),
        "4": (5.2e-08, 3.2768e-06),
        "5": (6.5536e-06, 6.5536e-06),
        "6": (1.31072e-05, 1.31104e-05),
}

MARGIN = 0.000


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the directory containing the output for each video",
    )
    return parser.parse_args()


def detect_event(morton_codes) -> list[int]:
    """
    Returns the event window frame interval, e.g. [140, 175]
    In case there is no event, returns [-1, -1]
    """

    temp_dict = CELL_MORTON.copy()
    event_window = []

    for index, row in morton_codes.iterrows():
        matching_cell_key = next((cell for cell, (min_value, max_value) in temp_dict.items()
                                       if min_value - MARGIN <= row["morton"] <= max_value + MARGIN), None)

        # print(f"Frame: {row['frame_id']}, Morton: {row['morton']}, Matching cell: {matching_cell_key}")

        if matching_cell_key is not None:
            # start the event window
            event_window.append(row["frame_id"])

            # remove the cells before the matching one from the temp_dict
            # matching_index = list(temp_dict).index(matching_cell_key)
            # keys_to_delete = list(temp_dict)[:matching_index + 1]
            # for key in keys_to_delete:
            #     del temp_dict[key]
            pass
        else:
            event_window.append(row["frame_id"])
            if len(event_window) > 2: # should maybe be more than 2, and might instead be based on event window duration aka abs(start - end)
                start = event_window[0]
                end = event_window[-1]
                return [start, end]

            event_window = []
            temp_dict = CELL_MORTON.copy()

    return [-1, -1]


def main(data_path):
    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if not os.path.isfile(os.path.join(target_path, "morton_codes.csv")):
            tqdm_obj.write(f"Skipping {video_id}: morton_codes.csv does not exist at {target_path}")
            continue

        morton_codes = pd.read_csv(os.path.join(target_path, "morton_codes.csv"), sep=";")

        event_window = detect_event(morton_codes)

        print(f"Event window for {video_id}: {event_window}")

        # Save the event window somewhere

    print("detector.py completed.")


if __name__ == "__main__":
    args = parse_arguments()

    main(args.data_path)
