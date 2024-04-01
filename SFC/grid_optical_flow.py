import os
import csv
import cv2
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import helper

NR_FRAMES_MOVING_AVG = 4
SCALE = 10
THICKNESS = 3
COLORS = [
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 128, 255),  # Orange
    (128, 0, 255),  # Pink
    (0, 255, 128),  # Lime
    (255, 128, 0),  # Sky Blue
    (128, 128, 128),  # Gray
    (128, 0, 0),  # Maroon
    (128, 128, 0),  # Olive
    (0, 128, 0),  # Dark Green
    (128, 0, 128),  # Purple
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the folder containing the dataset, e.g. ./data/{dataset_name}",
    )
    parser.add_argument(
        "output_path",
        help="Path to the folder where the output will be saved, e.g. ./output/{dataset_name}",
    )
    parser.add_argument(
        "data_config_path",
        help="Path to dataset config yml file",
    )
    parser.add_argument(
        "event_config_path",
        help="Path to event config yml file",
    )
    parser.add_argument("--display_results", action=argparse.BooleanOptionalAction)
    return parser.parse_args()

def calculate_cell_values(
    frame,
    frame_gray,
    frame_gray_prev,
    cell_angles,
    cell_positions,
    angle_threshold,
):
    h, w = frame_gray.shape[:3]

    flow = cv2.calcOpticalFlowFarneback(frame_gray_prev, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    sum_vectors = np.array([0.0, 0.0])
    vector_sum = np.zeros(2)

    current_frame_differences = []

    for cell_index, (top_left, bottom_right) in enumerate(cell_positions):
        # Define the cell boundaries
        cell_start_x = top_left[0]
        cell_start_y = top_left[1]
        cell_end_x = bottom_right[0]
        cell_end_y = bottom_right[1]

        # Ensure the cell is within the image boundaries
        cell_end_x = min(cell_end_x, w)
        cell_end_y = min(cell_end_y, h)

        # Compute mean flow vector for the cell
        cell_flow = flow[cell_start_y:cell_end_y, cell_start_x:cell_end_x]
        mean_flow = np.mean(cell_flow, axis=(0, 1))

        # Compute the direction (angle) of the mean flow vector
        angle_radians = np.arctan2(mean_flow[1], mean_flow[0])
        angle_degrees = np.degrees(angle_radians)

        if angle_degrees < 0:
            angle_degrees += 360  # Normalize angle

        # Accumulate vectors
        vector_sum += mean_flow

        arrow_color = COLORS[cell_index % len(COLORS)]

        cell_angles[cell_index].append(angle_degrees)

        sum_vectors += mean_flow

        moving_avg = None

        # Calculate moving average of the last n angles
        if len(cell_angles[cell_index]) >= NR_FRAMES_MOVING_AVG:
            moving_avg = np.mean(
                cell_angles[cell_index][-NR_FRAMES_MOVING_AVG - 1 : -1]
            )

            if moving_avg < 0:
                moving_avg += 360  # Normalize angle

        # Normalize the angle difference to be between 0 and -180 degrees
        if moving_avg is not None:
            angle_diff = angle_degrees - moving_avg

            # Adjusts angle_diff to be between 0 and 180
            if -360 <= angle_diff < -180:
                angle_diff += 180
            else:
                angle_diff = 0

            # Clamp the angle difference if it's less than the threshold
            clamped_angle_diff = -angle_diff if (-angle_diff) >= angle_threshold else 0
            moving_avg_vector = angle_to_vector(moving_avg, SCALE)  # This variable is never used (?)

            if clamped_angle_diff != 0:
                # This means clamped_angle_diff is -angle_diff and meets the threshold condition
                # Draw a bounding box around the cell
                cv2.rectangle(
                    frame,
                    (cell_start_x, cell_start_y),
                    (cell_end_x, cell_end_y),
                    (0, 255, 0),
                    2,
                )

            current_frame_differences.append(clamped_angle_diff)
        else:
            current_frame_differences.append(0)

        # Draw the mean vector at the center of the cell
        center_x = (cell_start_x + cell_end_x) // 2
        center_y = (cell_start_y + cell_end_y) // 2
        end_x = center_x + int(mean_flow[0]) * SCALE
        end_y = center_y + int(mean_flow[1]) * SCALE
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), arrow_color, THICKNESS)

    return current_frame_differences

def angle_to_vector(angle_degrees, scale=1):
    """
    Convert an angle in degrees to a vector.
    """
    angle_radians = np.radians(angle_degrees)
    x = np.cos(angle_radians) * scale
    y = np.sin(angle_radians) * scale
    return (x, y)

def process_video(
    video_path,
    target_path,
    video_id,
    data_config,
    event_config,
) -> dict:
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()

    if not ret:
        print("Unable to read video")
        exit(1)

    out = cv2.VideoWriter(
        os.path.join(target_path, f"{video_id}.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        data_config["fps"],
        (frame.shape[1], frame.shape[0]),
    )

    frame_gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_number = 0
    angle_threshold = event_config["angle_threshold"]

    # Maps a frame number to a list of cell values
    angle_diff_map = {}

    cell_positions = helper.calculate_grid_cell_positions(frame, event_config)

    # Keep track of cell angles to calculate average cell angles
    nr_cells = len(cell_positions)
    cell_angles = {cell_index: [] for cell_index in range(nr_cells)}

    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Frame progress", leave=False) as pbar_frames:
        while ret:
            frame_number += 1

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            angle_diff_map[frame_number] = calculate_cell_values(
                frame,
                frame_gray,
                frame_gray_prev,
                cell_angles,
                cell_positions,
                angle_threshold,
            )
            frame_gray_prev = frame_gray

            helper.draw_grid(frame, cell_positions)

            helper.annotate_frame(
                frame,
                f"frame: {frame_number}. angle_threshold: {angle_threshold}",
                (10, 30),
            )

            out.write(frame)

            if args.display_results:
                cv2.imshow("Grid", frame)

            ret, frame = cap.read()

            if cv2.waitKey(30) & 0xFF == ord("q"):
                print("Interrupted by user")
                break

            pbar_frames.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return angle_diff_map


def main(data_path, output_path, data_config, event_config, display_results):
    os.makedirs(output_path, exist_ok=True)

    total_cells = helper.get_total_cells(event_config)

    for video_dir, video_id, tqdm_obj in helper.traverse_videos(data_path):
        video_path = os.path.join(video_dir, f"{video_id}.avi")
        target_path = os.path.join(output_path, video_id)

        os.makedirs(os.path.join(output_path, video_id), exist_ok=True)

        angle_diff_map = process_video(
            video_path,
            target_path,
            video_id,
            data_config,
            event_config,
        )

        helper.save_csv(angle_diff_map, target_path, event_config)
        helper.save_cell_value_subplots(angle_diff_map, target_path, display_results, total_cells, "Angle difference")
        helper.save_combined_plot(angle_diff_map, target_path, display_results, total_cells, "Angle difference")

    helper.save_config(output_path, data_config, event_config)


if __name__ == "__main__":
    args = parse_arguments()
    data_config = helper.load_yml(args.data_config_path)
    event_config = helper.load_yml(args.event_config_path)
    main(
        args.data_path,
        args.output_path,
        data_config,
        event_config,
        args.display_results,
    )
