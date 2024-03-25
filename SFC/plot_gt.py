import os
import yaml
import argparse
import matplotlib.pyplot as plt
import pandas as pd


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


def load_yml(file_path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def create_and_save_CSP_with_ground_truth_and_dots(df, ground_truth, output_path):
    frame_start = ground_truth["event_window"][0]
    frame_end = ground_truth["event_window"][1]

    group_A = []
    group_B = []
    x_A, y_A, x_B, y_B = (
        [],
        [],
        [],
        [],
    )  # Lists to store Morton codes and frame IDs for dots plotting

    for _, row in df.iterrows():
        morton = row["morton"]
        frame_id = row["frame_id"]
        if morton == 0:
            continue
        if frame_id in range(frame_start, frame_end):
            group_A.append(morton)
            x_A.append(morton)
            y_A.append(frame_id)
        else:
            group_B.append(morton)
            x_B.append(morton)
            y_B.append(frame_id)

    data = [group_A, group_B]
    data_colors = ["red", "lightgray"]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Morton")
    ax1.set_ylabel("Frequency")
    ax1.set_ylim((0, 1))
    ax1.eventplot(
        data, orientation="horizontal", colors=data_colors, lineoffsets=[0.5, 0.5]
    )

    # Additional ax2 for dot plots
    ax2 = ax1.twinx()
    ax2.scatter(x_A, y_A, s=5, color="red")  # Dots for group A in red
    ax2.scatter(x_B, y_B, s=5, color="lightgray")  # Dots for group B in lightgray
    ax2.set_ylabel("Frame Number")
    ax2.set_ylim(
        [min(y_A + y_B), max(y_A + y_B)]
    )  # Adjusting the y limits based on frame ID data

    plt.title(
        f"{output_path} \n Video ID: {ground_truth['id']}. Event window: {frame_start}-{frame_end}"
    )
    plt.savefig(os.path.join(output_path, "morton_codes_ground_truth_with_dots.png"))
    plt.close()


def get_ground_truth(ground_truth, video_id):
    for video in ground_truth:
        if video["id"] == video_id:
            return video
    return None


def main(data_path: str, ground_truth: dict):
    assert os.path.exists(data_path), "Path does not exist"

    video_id = os.path.basename(data_path)
    morton_codes_path = os.path.join(data_path, "morton_codes.csv")
    video_ground_truth = get_ground_truth(ground_truth, video_id)

    if os.path.isfile(morton_codes_path) and ground_truth:
        morton_codes_df = pd.read_csv(morton_codes_path, sep=";")
        create_and_save_CSP_with_ground_truth_and_dots(
            morton_codes_df, video_ground_truth, data_path
        )
    else:
        print(f"Morton codes or ground truth does not exist for video {video_id}")
        exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    ground_truth = load_yml(args.ground_truth_path)

    main(args.data_path, ground_truth)
