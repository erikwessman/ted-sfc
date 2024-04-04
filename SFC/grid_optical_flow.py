import os
import cv2
import argparse
import numpy as np
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

EVENT_ANGLE = 0
EVENT_ANGLE_RANGE = 10
FLOW_THRESHOLD = 7


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
    cell_positions,
    event_angle,
    event_angle_range,
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

        # Compute the direction (angle) of the mean flow vector
        angle_radians = np.arctan2(cell_mean_flow[1], cell_mean_flow[0])
        angle_degrees = np.degrees(angle_radians)

        if angle_degrees < 0:
            angle_degrees += 360  # Normalize angle

        distance_to_event_angle = abs(angle_degrees - event_angle)

        is_event_cell = (
            distance_to_event_angle < event_angle_range
            and np.linalg.norm(cell_mean_flow) > flow_threshold
        )

        cell_value = 1 - (distance_to_event_angle / 360) if is_event_cell else 0

        frame_cell_values.append(cell_value)

        if is_event_cell:
            # Draw a bounding box around the cell
            cv2.rectangle(
                frame,
                (cell_start_x, cell_start_y),
                (cell_end_x, cell_end_y),
                (0, 255, 0),
                2,
            )

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
    data_config,
    event_config,
) -> dict:
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()

    if not ret:
        print("Unable to read video")
        exit(1)

    out = cv2.VideoWriter(
        os.path.join(target_path, f"{video_id}_optical_flow_grid.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        data_config["fps"],
        (frame.shape[1], frame.shape[0]),
    )

    frame_gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_number = 0

    cell_positions = helper.calculate_grid_cell_positions(frame, event_config)

    # Maps a frame number to a list containing the cell values for that frame which match the event criteria
    cell_values = {}

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
                EVENT_ANGLE,
                EVENT_ANGLE_RANGE,
                FLOW_THRESHOLD,
            )
            frame_gray_prev = frame_gray

            helper.draw_grid(frame, cell_positions)

            helper.annotate_frame(
                frame,
                f"frame: {frame_number}. angle range: {EVENT_ANGLE_RANGE}, flow threshold: {FLOW_THRESHOLD}",
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

    return cell_values


def main(data_path, output_path, data_config, event_config, display_results):
    os.makedirs(output_path, exist_ok=True)

    for video_dir, video_id, tqdm_obj in helper.traverse_videos(data_path):
        video_path = os.path.join(video_dir, f"{video_id}.avi")
        target_path = os.path.join(output_path, video_id)

        os.makedirs(os.path.join(output_path, video_id), exist_ok=True)

        output_cell_value_map = process_video(
            video_path,
            target_path,
            video_id,
            data_config,
            event_config,
        )

        helper.save_cell_value_csv(output_cell_value_map, target_path, event_config)
        helper.save_cell_value_subplots(
            output_cell_value_map, target_path, display_results, "Angle difference"
        )
        helper.save_combined_plot(
            output_cell_value_map, target_path, display_results, "Angle difference"
        )

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
