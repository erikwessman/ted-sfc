import os
import argparse
import pandas as pd

import helper


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the directory containing the output for each video",
    )
    parser.add_argument(
        "search_mask_path",
        help="Path to the directory containing the output for each video",
    )
    return parser.parse_args()


def main(data_path, search_mask):
    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if os.path.isfile(os.path.join(target_path, "cell_values.csv")):
            tqdm_obj.write(f"Skipping {video_id}: Cell values CSV does not exist")
            continue

        cell_values = pd.read_csv(os.path.join(target_path, "cell_values.csv"), sep=";")

    print("detector.py completed.")


if __name__ == "__main__":
    args = parse_arguments()

    main(args.data_path, args.search_mask_path)
