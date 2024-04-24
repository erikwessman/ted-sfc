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
        "ground_truth_path",
        help="Path to the ground truth YML file.",
    )
    return parser.parse_args()


def main(data_path: str, ground_truth_path: str):
    ground_truth = helper.load_yml(ground_truth_path)

    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        # cell_values_path = os.path.join(video_path, "cell_values.csv")
        morton_codes_path = os.path.join(video_path, "morton_codes.csv")

        # if not os.path.exists(cell_values_path):
        #     tqdm_obj.write(f"Skipping {video_id}: Cell values CSV does not exist")
        #     continue

        if not os.path.exists(morton_codes_path):
            tqdm_obj.write(f"Skipping {video_id}: Morton codes CSV does not exist")
            continue

        video_ground_truth = helper.get_ground_truth(ground_truth, video_id)

        if not video_ground_truth:
            tqdm_obj.write(f"Skipping {video_id}: Ground truth does not exist")
            continue

        morton_codes_df = pd.read_csv(morton_codes_path, sep=";")
        # cell_values_df = pd.read_csv(cell_values_path, sep=";")

        plot_path = os.path.join(video_path, "ground_truth_plots")
        os.makedirs(plot_path, exist_ok=True)

        event_window = video_ground_truth["event_window"]
        helper.create_and_save_CSP(morton_codes_df, plot_path, False, event_window)
        # helper.save_cell_value_subplots(
        #     cell_values_df, plot_path, False, "Cell Value", event_window
        # )


if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path, args.ground_truth_path)
