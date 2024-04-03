import os
import pandas as pd
import argparse

import helper


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate F1 scores based on ground truth and prediction data.")
    parser.add_argument("event_window_path", type=str, help="Path to event_window.csv")
    parser.add_argument("ground_truth_path", type=str, help="Path to the ground truth YML file")
    parser.add_argument("event_config_path", type=str, help="Path to the event config YML file")
    return parser.parse_args()


def is_overlapping(true_window, pred_window):
    true_start, true_end = true_window
    pred_start, pred_end = pred_window
    overlap = max(0, min(true_end, pred_end) - max(true_start, pred_start))
    return overlap > 0


def main(event_window_path: str, ground_truth: dict, config: dict):
    TP, FP, FN, TN = 0, 0, 0, 0

    TP_videos, FP_videos, FN_videos, TN_videos = [], [], [], []

    assert os.path.exists(event_window_path), "Event window file does not exist"

    df_event_window = pd.read_csv(event_window_path, sep=";")
    event_direction = config["direction"]

    for index, row in df_event_window.iterrows():
        video_id = row["video_id"]
        start_frame = row["start_frame"]
        end_frame = row["end_frame"]
        event_detected = row["event_detected"]

        video_ground_truth = helper.get_ground_truth(ground_truth, video_id, event_direction)

        if event_detected:
            if video_ground_truth:
                true_start, true_end = video_ground_truth["event_window"]
                if is_overlapping((true_start, true_end), (start_frame, end_frame)):
                    TP += 1
                    TP_videos.append(video_id)
                else:
                    FP += 1
                    FP_videos.append(video_id)
            else:
                FP += 1
                FP_videos.append(video_id)
        else:
            if video_ground_truth:
                FN += 1
                FN_videos.append(video_id)
            else:
                TN += 1
                TN_videos.append(video_id)

    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"TP: {TP_videos}\n, FP: {FP_videos}\n, FN: {FN_videos}\n, TN: {TN_videos}\n")

    f1_score = 2 * TP / (2 * TP + FP + FN)

    print(f"F1: {f1_score}")


if __name__ == "__main__":
    args = parse_arguments()
    ground_truth = helper.load_yml(args.ground_truth_path)
    config = helper.load_yml(args.event_config_path)

    main(args.event_window_path, ground_truth, config)
