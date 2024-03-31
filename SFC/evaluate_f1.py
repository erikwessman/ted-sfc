import os
import argparse
from sklearn.metrics import precision_recall_fscore_support

import helper


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate F1 scores based on ground truth and prediction data.")
    parser.add_argument("data_path", type=str, help="Path to the root folder containing prediction subfolders")
    parser.add_argument("ground_truth_path", type=str, help="Path to the ground truth YML file")
    parser.add_argument("event_config_path", type=str, help="Path to the event config YML file")
    return parser.parse_args()


def is_overlapping(true_window, pred_window):
    true_start, true_end = true_window
    pred_start, pred_end = pred_window
    overlap = max(0, min(true_end, pred_end) - max(true_start, pred_start))
    return overlap > 0


def main(data_path: str, ground_truth: dict, config: dict):
    TP, FP, FN = 0, 0, 0

    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        prediction_path = os.path.join(video_path, "predicted_event_window.yml")

        if not prediction_path.exists():
            tqdm_obj.write(f"Skipping {video_id}: Predicted event window does not exist")
            continue

        prediction = helper.load_yml(prediction_path)
        event_direction = config["direction"]
        video_ground_truth = helper.get_ground_truth(ground_truth, video_id, event_direction)

        if not video_ground_truth:
            tqdm_obj.write(f"Skipping {video_id}: Ground truth does not exist")
            continue

        if is_overlapping(ground_truth['event_window'], prediction['event_window']):
            TP += 1
        else:
            FN += 1

        # TODO fix this
        if TP == 0:
            FP += 1

    precision, recall, f1_score, _ = precision_recall_fscore_support([1]*TP + [0]*FN, [1]*TP + [0]*FP, average='binary')

    output_path = os.path.join(data_path, "evaluation.txt")
    with open(output_path, "w") as f:
        f.write(f'F1 Score: {f1_score}\n')
        f.write(f'Precision Score: {precision}\n')
        f.write(f'Recall Score: {recall}\n')

    print("evaluate_f1.py completed")


if __name__ == "__main__":
    args = parse_arguments()
    ground_truth = helper.load_yml(args.ground_truth_path)
    config = helper.load_yml(args.event_config_path)

    main(args.data_path, ground_truth, config)
