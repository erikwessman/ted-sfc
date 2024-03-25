import os
import yaml
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate F1 scores based on ground truth and prediction data.")
    parser.add_argument("ground_truth_path", type=str, help="Path to the ground truth YML file")
    parser.add_argument("predictions_root_path", type=str, help="Path to the root folder containing prediction subfolders")
    return parser.parse_args()


def load_yml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def calculate_overlap(gt_window, pred_start, pred_end):
    gt_start, gt_end = gt_window
    overlap = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    return overlap > 0


def main(data_path: str, ground_truth: dict):
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
        prediction_path = os.path.join(target_path, "predicted_event_window.yml")

        if prediction_path.exists():
            prediction = load_yml(prediction_path)

            # Initialize counters
            TP, FP, FN = 0, 0, 0

            gt_events = [item for item in ground_truth if item['id'] == video_id]
            if not gt_events:
                FP += 1
            else:
                for gt_event in gt_events:
                    if calculate_overlap(gt_event['event_window'], prediction['event_start'], prediction['event_end']):
                        TP += 1
                    else:
                        FN += 1
                if TP == 0:
                    FP += 1

            precision, recall, f1_score, _ = precision_recall_fscore_support([1]*TP + [0]*FN, [1]*TP + [0]*FP, average='binary')

            # Save the F1 score
            with open("evaluation.txt", "w") as f:
                f.write(f'F1 Score: {f1_score}\n')
                f.write(f'Precision Score: {precision}\n')
                f.write(f'Recall Score: {recall}\n')
        else:
            print(f"Prediction for {video_id} does not exist, skipping")

        pbar.set_description("Processing folders")


if __name__ == "__main__":
    args = parse_arguments()
    ground_truth = load_yml(args.ground_truth_path)

    main(args.data_path, ground_truth)
