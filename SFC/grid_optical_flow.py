import os
import csv
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

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


def draw_grid(frame, cell_positions, line_color=(255, 255, 255), line_thickness=1):
    for cell_index, (top_left, bottom_right) in enumerate(cell_positions):
        start_x, start_y = top_left
        end_x, end_y = bottom_right

        # Draw the top line of the cell
        cv2.line(
            frame,
            (start_x, start_y),
            (end_x, start_y),
            color=line_color,
            thickness=line_thickness,
        )
        # Draw the bottom line of the cell
        cv2.line(
            frame,
            (start_x, end_y),
            (end_x, end_y),
            color=line_color,
            thickness=line_thickness,
        )
        # Draw the left line of the cell
        cv2.line(
            frame,
            (start_x, start_y),
            (start_x, end_y),
            color=line_color,
            thickness=line_thickness,
        )
        # Draw the right line of the cell
        cv2.line(
            frame,
            (end_x, start_y),
            (end_x, end_y),
            color=line_color,
            thickness=line_thickness,
        )

        # Draw the cell index in the top-left corner
        cv2.putText(
            frame,
            str(cell_index + 1),
            (start_x + 20, start_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )


def calculate_cell_positions(image, config):
    h, w, _ = image.shape

    # Convert proportional positions to pixel positions
    start_x = int(w * config["grid_top_left"][0])
    start_y = int(h * config["grid_top_left"][1])
    end_x = int(w * config["grid_bottom_right"][0])
    end_y = int(h * config["grid_bottom_right"][1])

    cell_width = (end_x - start_x) // config["grid_num_cols"]
    cell_height = (end_y - start_y) // config["grid_num_rows"]

    cell_positions = []

    if config["grid_direction"] == "right":
        col_range = range(config["grid_num_cols"])
    else:
        col_range = range(config["grid_num_cols"] - 1, -1, -1)

    for row in range(config["grid_num_rows"]):
        for col in col_range:
            cell_start_x = start_x + col * cell_width
            cell_start_y = start_y + row * cell_height

            cell_end_x = cell_start_x + cell_width
            cell_end_y = cell_start_y + cell_height

            cell_positions.append(
                ((cell_start_x, cell_start_y), (cell_end_x, cell_end_y))
            )

    return cell_positions


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


def annotate_frame(
    frame, text, position, font_scale=1, font_color=(255, 255, 255), thickness=2
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)


def angle_to_vector(angle_degrees, scale=1):
    """
    Convert an angle in degrees to a vector.
    """
    angle_radians = np.radians(angle_degrees)
    x = np.cos(angle_radians) * scale
    y = np.sin(angle_radians) * scale
    return (x, y)


def save_csv(angle_diff_map, output_path, config):
    csv_path = os.path.join(output_path, "cell_values.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        number_of_cells = config["grid_num_rows"] * config["grid_num_cols"]
        writer.writerow(["frame_id"] + [f"cell{i+1}" for i in range(number_of_cells)])
        for key, values in angle_diff_map.items():
            row = [key] + values
            writer.writerow(row)


def save_config(output_path, data_config, event_config):
    config_path = os.path.join(output_path, "config.txt")
    with open(config_path, "w") as f:
        for key, value in data_config.items():
            line = f"{key}: {value}\n"
            f.write(line)
        for key, value in event_config.items():
            line = f"{key}: {value}\n"
            f.write(line)


def save_plots(angle_diff_map, output_path, display_results, event_config):
    fig, axs = plt.subplots(
        nrows=event_config["grid_num_rows"],
        ncols=event_config["grid_num_cols"],
        figsize=(event_config["grid_num_cols"] * 4, event_config["grid_num_rows"] * 3),
    )

    for cell_index, ax in enumerate(axs.flat):
        ax.plot([value[cell_index] for value in angle_diff_map.values()], label=f"Cell {cell_index+1}",)
        ax.set_title(f"Cell {cell_index+1}")

        if cell_index == 0:
            ax.legend()

    plt.legend()

    fig.text(0.5, 0.04, "Frame", ha="center")
    fig.text(0.04, 0.5, "Angle Differences", va="center", rotation="vertical")

    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.97, top=0.95, wspace=0.2, hspace=0.4)

    plt.savefig(os.path.join(output_path, "all_cells.png"))

    if display_results:
        plt.show()
    else:
        plt.close()


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

    cell_positions = calculate_cell_positions(frame, event_config)

    # Keep track of cell angles to calculate average cell angles
    nr_cells = len(cell_positions)
    cell_angles = {cell_index: [] for cell_index in range(nr_cells)}

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

        draw_grid(frame, cell_positions)

        annotate_frame(
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

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return angle_diff_map


def main(data_path, output_path, data_config, event_config, display_results):
    os.makedirs(output_path, exist_ok=True)

    for video_path, video_id, tqdm in helper.traverse_videos(data_path):
        target_path = os.path.join(output_path, video_id)

        os.makedirs(os.path.join(output_path, video_id), exist_ok=True)

        angle_diff_map = process_video(
            video_path,
            target_path,
            video_id,
            data_config,
            event_config,
        )

        save_csv(angle_diff_map, target_path, event_config)
        save_plots(angle_diff_map, target_path, display_results, event_config)

    save_config(output_path, data_config, event_config)


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
