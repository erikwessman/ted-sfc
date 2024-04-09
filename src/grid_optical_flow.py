import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import helper

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

EVENT_ANGLES = [0, 180]
DEFAULT_ANGLE = 90
ANGLE_RANGE_THRESHOLD = 30
FLOW_THRESHOLD = 5
ANGLE_DIFF_THRESHOLD = 50
NR_FRAMES_MOVING_AVG = 10


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
        "config_path",
        help="Path to the config yml file",
    )
    parser.add_argument("--display_results", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def check_event_criteria(
    angle_distance,
    angle_diff,
    angle_range_threshold,
    angle_diff_threshold,
):
    angle_range_criterion = angle_distance < angle_range_threshold / 2
    angle_diff_criterion = angle_diff >= angle_diff_threshold

    return angle_range_criterion and angle_diff_criterion


def angle_difference(angle1, angle2):
    return abs((angle2 - angle1 + 180) % 360 - 180)


def calculate_cell_values(
    frame,
    frame_gray,
    frame_gray_prev,
    cell_positions,
    moving_avg_cell_angles,
    event_angles,
    angle_range_threshold,
    angle_diff_threshold,
    flow_threshold,
):
    h, w = frame_gray.shape[:3]

    flow = cv2.calcOpticalFlowFarneback(
        frame_gray_prev, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    frame_cell_values = []

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
        cell_mean_flow = np.mean(cell_flow, axis=(0, 1))

        if np.linalg.norm(cell_mean_flow) <= flow_threshold:
            current_angle = DEFAULT_ANGLE
        else:
            # Compute the direction (angle) of the mean flow vector
            angle_radians = np.arctan2(cell_mean_flow[1], cell_mean_flow[0])
            current_angle = np.degrees(angle_radians)

        moving_avg_cell_angles[cell_index].append(current_angle)

        # Calculate moving average of the last n angles, excluding the current one
        moving_avg = None
        if len(moving_avg_cell_angles[cell_index]) >= NR_FRAMES_MOVING_AVG:
            moving_avg = np.mean(
                moving_avg_cell_angles[cell_index][-NR_FRAMES_MOVING_AVG - 1 : -1]
            )

            if moving_avg < 0:
                moving_avg += 360  # Normalize angle

        if moving_avg is None:
            frame_cell_values.append(0)
            continue

        # Calculate the difference between the current angle and the moving average
        angle_diff = angle_difference(current_angle, moving_avg)

        cell_value = 0
        for event_angle in event_angles:
            distance_to_event_angle = angle_difference(event_angle, current_angle)

            is_event_cell = check_event_criteria(
                distance_to_event_angle,
                angle_diff,
                angle_range_threshold,
                angle_diff_threshold,
            )

            if is_event_cell:
                cell_value = 1 - (distance_to_event_angle / 360)

                # Draw a bounding box around the cell
                cv2.rectangle(
                    frame,
                    (cell_start_x, cell_start_y),
                    (cell_end_x, cell_end_y),
                    (0, 255, 0),
                    2,
                )

                break
            else:
                cell_value = 0

        frame_cell_values.append(cell_value)

        # Draw the mean vector at the center of the cell
        center_x = (cell_start_x + cell_end_x) // 2
        center_y = (cell_start_y + cell_end_y) // 2
        end_x = center_x + int(cell_mean_flow[0]) * SCALE
        end_y = center_y + int(cell_mean_flow[1]) * SCALE
        arrow_color = COLORS[cell_index % len(COLORS)]
        cv2.arrowedLine(
            frame, (center_x, center_y), (end_x, end_y), arrow_color, THICKNESS
        )

    return frame_cell_values


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
    config,
) -> dict:
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()

    if not ret:
        print("Unable to read video")
        exit(1)

    out = cv2.VideoWriter(
        os.path.join(target_path, f"{video_id}_optical_flow_grid.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        config["fps"],
        (frame.shape[1], frame.shape[0]),
    )

    frame_gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_number = 0

    cell_positions = helper.calculate_grid_cell_positions(frame, config)

    # Maps a frame number to a list containing the cell values for that frame which match the event criteria
    cell_values = {}

    # Keep track of cell angles to calculate average cell angles
    moving_avg_cell_angles = {
        cell_index: [] for cell_index in range(len(cell_positions))
    }

    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Frame progress", leave=False) as pbar_frames:
        while ret:
            frame_number += 1

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cell_values[frame_number] = calculate_cell_values(
                frame,
                frame_gray,
                frame_gray_prev,
                cell_positions,
                moving_avg_cell_angles,
                EVENT_ANGLES,
                ANGLE_RANGE_THRESHOLD,
                ANGLE_DIFF_THRESHOLD,
                FLOW_THRESHOLD,
            )
            frame_gray_prev = frame_gray

            helper.draw_grid(frame, cell_positions)

            helper.annotate_frame(
                frame,
                f"frame: {frame_number}, event_angles: {EVENT_ANGLES}, angle_range: {ANGLE_RANGE_THRESHOLD}, ",
                (10, 30),
            )
            helper.annotate_frame(
                frame,
                f"angle_diff_threshold: {ANGLE_DIFF_THRESHOLD}, flow_threshold: {FLOW_THRESHOLD}",
                (10, 60),
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

    return cell_values


def main(data_path, output_path, grid_config, display_results):
    os.makedirs(output_path, exist_ok=True)

    for video_dir, video_id, tqdm_obj in helper.traverse_videos(data_path):
        video_path = os.path.join(video_dir, f"{video_id}.avi")
        target_path = os.path.join(output_path, video_id)

        os.makedirs(os.path.join(output_path, video_id), exist_ok=True)

        output_cell_value_map = process_video(
            video_path,
            target_path,
            video_id,
            grid_config,
        )

        helper.save_cell_value_csv(output_cell_value_map, target_path, grid_config)
        helper.save_cell_value_subplots(
            output_cell_value_map, target_path, display_results, "Cell value"
        )
        helper.save_combined_plot(
            output_cell_value_map, target_path, display_results, "Cell value"
        )

    helper.save_config(grid_config, output_path, "grid_config.yml")
    print("grid_optical_flow.py completed.")


if __name__ == "__main__":
    args = parse_arguments()
    config = helper.load_yml(args.config_path)
    grid_config = config["grid_config"]

    main(
        args.data_path,
        args.output_path,
        grid_config,
        args.display_results,
    )
