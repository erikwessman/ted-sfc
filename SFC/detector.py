import os
import argparse
import pandas as pd
from tqdm import tqdm


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


def main(data_path, display_plots):
    assert os.path.exists(data_path), f"Data path {data_path} does not exist."

    video_dirs = [
        name
        for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    ]

    pbar = tqdm(video_dirs, desc="Processing folders")
    for video_id in pbar:
        pbar.set_description(f"Processing folder {video_id}")
        target_path = os.path.join(data_path, video_id)

        if os.path.isfile(os.path.join(target_path, "cell_values.csv")):
            cell_values = pd.read_csv(
                os.path.join(target_path, "cell_values.csv"), sep=";"
            )
        else:
            tqdm.write(f"Skipped. File 'cell_values.csv' not found in {target_path}.")

        pbar.set_description("Processing folders")

    print("morton.py completed.")


if __name__ == "__main__":
    args = parse_arguments()

    main(args.data_path, args.search_mask_path)
