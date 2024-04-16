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
    parser.add_argument(
        "--cpu", help="Use CPU instead of GPU", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--display_results", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def check_event_criteria(
    angle_distance,
    angle_diff,
    mean_flow,
    angle_range_threshold,
    angle_diff_threshold,
    flow_threshold,
):
    angle_range_criterion = angle_distance < angle_range_threshold / 2
    angle_diff_criterion = angle_diff >= angle_diff_threshold
    flow_criterion = np.linalg.norm(mean_flow) >= flow_threshold

    return angle_range_criterion and angle_diff_criterion and flow_criterion


def vector_angle_difference(vector1, vector2):
    cos_theta = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def circular_angle_distance(angle1, angle2):
    return min((angle1 - angle2) % 360, (angle2 - angle1) % 360)


def calculate_cuda_optical_flow(frame_gray_prev, frame_gray):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise Exception(
            "No CUDA device found. This code requires a CUDA-enabled GPU and OpenCV with CUDA support."
        )

    cuda_farneback = cv2.cuda_FarnebackOpticalFlow.create(
        numLevels=5,
        pyrScale=0.5,
        fastPyramids=False,
        winSize=15,
        numIters=5,
        polyN=5,
        polySigma=1.1,
        flags=0,
    )

    # Upload images to GPU memory
    prev_gpu = cv2.cuda_GpuMat(frame_gray_prev)
    curr_gpu = cv2.cuda_GpuMat(frame_gray)

    flow_gpu = cuda_farneback.calc(prev_gpu, curr_gpu, None)

    return flow_gpu.download()


def calculate_cell_values(
    frame,
    frame_gray,
    frame_gray_prev,
    cell_positions,
    moving_avg_cell_angles,
    use_cpu,
    optical_flow_config,
):
    h, w = frame_gray.shape[:3]

    if use_cpu:
        flow = cv2.calcOpticalFlowFarneback(
            frame_gray_prev, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
    else:
        flow = calculate_cuda_optical_flow(frame_gray_prev, frame_gray)

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
        cell_flow_vector = np.mean(cell_flow, axis=(0, 1))

        # HACKY SOLUTION
        # If the optical flow is small, we add a large vector straight down to indicate standing still
        if np.linalg.norm(cell_flow_vector) <= 1:
            moving_avg_cell_angles[cell_index].append(np.array([0, 100]))
        else:
            moving_avg_cell_angles[cell_index].append(cell_flow_vector)

        if (
            len(moving_avg_cell_angles[cell_index])
            >= optical_flow_config["nr_frames_moving_avg"]
        ):
            moving_avg = np.array(
                moving_avg_cell_angles[cell_index][
                    -optical_flow_config["nr_frames_moving_avg"] - 1 : -3
                ]
            )
            moving_avg_vector = np.mean(moving_avg, axis=0)
            moving_avg_angle = np.degrees(
                np.arctan2(moving_avg_vector[0], moving_avg_vector[1])
            )

            current_3 = np.array(moving_avg_cell_angles[cell_index][-3:])
            current_3_vector = np.mean(current_3, axis=0)
            current_3_angle = np.degrees(
                np.arctan2(current_3_vector[0], current_3_vector[1])
            )
        else:
            frame_cell_values.append(0)
            continue

        # Calculate the difference between the current angle and the moving average
        vector_angle_diff = vector_angle_difference(current_3_vector, moving_avg_vector)
        # vector_magnitude_diff = np.linalg.norm(current_3_vector - moving_avg_vector)

        cell_value = 0
        for event_angle in optical_flow_config["event_angles"]:
            distance_to_event_angle = circular_angle_distance(
                event_angle, current_3_angle
            )

            is_event_cell = check_event_criteria(
                distance_to_event_angle,
                vector_angle_diff,
                cell_flow_vector,
                optical_flow_config["angle_range_threshold"],
                optical_flow_config["angle_diff_threshold"],
                optical_flow_config["flow_threshold"],
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

        text_scale = 0.5
        text_thickness = 1

        if cell_index == 0:
            cv2.putText(
                frame,
                "Moving avg angle",
                (cell_start_x - 200, cell_end_y + 20),  # Adjust text position as needed
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
            )
            cv2.putText(
                frame,
                "Current 3 angle",
                (cell_start_x - 200, cell_end_y + 40),  # Adjust text position as needed
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
            )
            cv2.putText(
                frame,
                "Angle difference",
                (cell_start_x - 200, cell_end_y + 60),  # Adjust text position as needed
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
            )

        cv2.putText(
            frame,
            f"{moving_avg_angle:.2f}",
            (cell_start_x, cell_end_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
        )

        cv2.putText(
            frame,
            f"{current_3_angle:.2f}",
            (cell_start_x, cell_end_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
        )

        cv2.putText(
            frame,
            f"{vector_angle_diff:.2f}",
            (cell_start_x, cell_end_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
        )

        center_x = (cell_start_x + cell_end_x) // 2
        center_y = (cell_start_y + cell_end_y) // 2
        end_x = center_x + int(cell_flow_vector[0]) * SCALE
        end_y = center_y + int(cell_flow_vector[1]) * SCALE
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
    video_path, target_path, video_id, config, display_results, use_cpu
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

    optical_flow_config = config["optical_flow"]

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
                use_cpu,
                optical_flow_config,
            )
            frame_gray_prev = frame_gray

            helper.draw_grid(frame, cell_positions)

            helper.annotate_frame(
                frame,
                f"frame: {frame_number}, event_angles: {optical_flow_config['event_angles']}, angle_range: {optical_flow_config['angle_range_threshold']}",
                (10, 30),
            )
            helper.annotate_frame(
                frame,
                f"angle_diff_threshold: {optical_flow_config['angle_diff_threshold']}, flow_threshold: {optical_flow_config['flow_threshold']}",
                (10, 60),
            )

            out.write(frame)

            if display_results:
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


def main(
    data_path: str,
    output_path: str,
    config_path: str,
    display_results: bool = False,
    use_cpu: bool = False,
):
    # Load config
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]

    os.makedirs(output_path, exist_ok=True)

    for video_dir, video_id, tqdm_obj in helper.traverse_videos(data_path):
        video_path = os.path.join(video_dir, f"{video_id}.avi")
        target_path = os.path.join(output_path, video_id)

        os.makedirs(os.path.join(output_path, video_id), exist_ok=True)

        output_cell_value_map = process_video(
            video_path, target_path, video_id, grid_config, display_results, use_cpu
        )

        helper.save_cell_value_csv(output_cell_value_map, target_path, grid_config)
        helper.save_cell_value_subplots(
            output_cell_value_map, target_path, display_results, "Cell value"
        )
        helper.save_combined_plot(
            output_cell_value_map, target_path, display_results, "Cell value"
        )

    helper.save_config(grid_config, output_path, "grid_config.yml")


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.data_path,
        args.output_path,
        args.config_path,
        args.display_results,
        args.cpu,
    )
