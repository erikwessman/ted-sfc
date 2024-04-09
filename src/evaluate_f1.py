import os
import pandas as pd
import argparse

import helper


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate F1 scores based on ground truth and prediction data."
    )
    parser.add_argument("event_window_path", type=str, help="Path to event_window.csv")
    parser.add_argument(
        "ground_truth_path", type=str, help="Path to the ground truth YML file"
    )
    return parser.parse_args()


def is_overlapping(true_window, pred_window):
    true_start, true_end = true_window
    pred_start, pred_end = pred_window
    overlap = max(0, min(true_end, pred_end) - max(true_start, pred_start))
    return overlap > 0


def calculate_iou(prediction_interval, ground_truth_interval):
    prediction_start, prediction_end = prediction_interval
    ground_truth_start, ground_truth_end = ground_truth_interval

    intersection_start = max(prediction_start, ground_truth_start)
    intersection_end = min(prediction_end, ground_truth_end)
    intersection = max(0, intersection_end - intersection_start)

    total_start = min(prediction_start, ground_truth_start)
    total_end = max(prediction_end, ground_truth_end)
    union = (
        (total_end - total_start)
        - intersection
        + (intersection_end - intersection_start)
    )

    iou = intersection / union if union != 0 else 0
    return iou


def main(event_window_path: str, ground_truth_path: str):
    # Load ground truth
    ground_truth = helper.load_yml(ground_truth_path)

    assert os.path.exists(event_window_path), "Event window file does not exist"

    df_event_window = pd.read_csv(event_window_path, dtype={"video_id": str}, sep=";")

    TP, FP, FN, TN = 0, 0, 0, 0

    iou_map = {}

    for index, row in df_event_window.iterrows():
        video_id = row["video_id"]

        event_detected = row["event_detected"]
        prediction_interval = (row["start_frame"], row["end_frame"])
        video_ground_truth = helper.get_ground_truth(ground_truth, video_id)

        if event_detected:
            if video_ground_truth:
                ground_truth_interval = video_ground_truth["event_window"]
                iou_score = calculate_iou(ground_truth_interval, prediction_interval)

                iou_map[video_id] = iou_score

                if iou_score:
                    TP += 1
                else:
                    FP += 1
            else:
                FP += 1
        else:
            if video_ground_truth:
                FN += 1
            else:
                TN += 1

    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

    f1_score = 2 * TP / (2 * TP + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    mean_iou = sum(iou_map.values()) / len(iou_map)

    print(f"F1: {f1_score}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Mean IoU: {mean_iou}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.event_window_path, args.ground_truth_path)
