import os
import csv
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm as tqdm

import helper


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


def ensure_matching_video_resolution(original_video_path: str, target_video_path: str):
    # Open the original video and get its resolution
    original_cap = cv2.VideoCapture(original_video_path)
    original_width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_cap.release()

    # Open the new video and get its resolution
    target_cap = cv2.VideoCapture(target_video_path)
    target_width = int(target_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(target_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if the resolutions are the same
    if original_width == target_width and original_height == target_height:
        target_cap.release()
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = target_cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(
            target_video_path, fourcc, fps, (original_width, original_height)
        )

        # Read through the new video, resize each frame, and write to the output
        while target_cap.isOpened():
            ret, frame = target_cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (original_width, original_height))
            out.write(resized_frame)

        target_cap.release()
        out.release()
        print(f"Target has been resized and saved to {target_video_path}.")


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


def calculate_grid_cell_positions(image, grid_config):
    h, w, _ = image.shape
    all_cell_positions = []
    grid_direction = grid_config["direction"]

    for grid_config in grid_config["grids"]:
        # Convert proportional positions to pixel positions for each grid
        start_x = int(w * grid_config["top_left"][0])
        start_y = int(h * grid_config["top_left"][1])
        end_x = int(w * grid_config["bottom_right"][0])
        end_y = int(h * grid_config["bottom_right"][1])

        cell_width = (end_x - start_x) // grid_config["cols"]
        cell_height = (end_y - start_y) // grid_config["rows"]

        cell_positions = []

        if grid_direction == "right":
            col_range = range(grid_config["cols"])
        else:
            col_range = range(grid_config["cols"] - 1, -1, -1)

        for row in range(grid_config["rows"]):
            for col in col_range:
                cell_start_x = start_x + col * cell_width
                cell_start_y = start_y + row * cell_height

                cell_end_x = cell_start_x + cell_width
                cell_end_y = cell_start_y + cell_height

                cell_positions.append(
                    ((cell_start_x, cell_start_y), (cell_end_x, cell_end_y))
                )

        all_cell_positions.extend(cell_positions)

    return all_cell_positions


def calculate_cell_values(frame, cell_positions, saliency_threshold):
    h, w = frame.shape[:2]

    heatmap_mean_values = []

    for top_left, bottom_right in cell_positions:
        # Define the cell boundaries
        cell_start_x = top_left[0]
        cell_start_y = top_left[1]
        cell_end_x = bottom_right[0]
        cell_end_y = bottom_right[1]

        # Ensure the cell is within the image boundaries
        cell_end_x = min(cell_end_x, w)
        cell_end_y = min(cell_end_y, h)

        # Calculate the mean value of the cell region
        cell_region = frame[cell_start_y:cell_end_y, cell_start_x:cell_end_x]
        # normalize cell_mean_value to be between 0 and 1
        cell_mean_value = np.mean(cell_region) / 255

        # Get the value if its over the SALIENCY_THRESHOLD, otherwise 0
        cell_mean_value = (cell_mean_value if cell_mean_value >= saliency_threshold else 0)

        heatmap_mean_values.append(cell_mean_value)

        # Draw a bounding box around the cell if the value is above the SALIENCY_THRESHOLD
        if cell_mean_value > saliency_threshold:
            cv2.rectangle(
                frame,
                (cell_start_x, cell_start_y),
                (cell_end_x, cell_end_y),
                (0, 255, 0),
                5,
            )

    return heatmap_mean_values


def overlay_heatmap(frame_heatmap, frame_original):
    heatmap = cv2.applyColorMap(
        (frame_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(frame_original, 0.6, heatmap, 0.4, 0)


def annotate_frame(
    frame, text, position, font_scale=1, font_color=(255, 255, 255), thickness=2
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)


def save_csv(mean_attention_map, output_path, grids_config):
    csv_path = os.path.join(output_path, "cell_values.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")

        # Calculate the total number of cells based on the individual grids
        cell_headers = []
        cell_counter = 0
        for grid in grids_config['grids']:
            cells_in_grid = grid["rows"] * grid["cols"]
            cell_headers.extend([f"cell{cell_counter + i + 1}" for i in range(cells_in_grid)])
            cell_counter += cells_in_grid

        writer.writerow(["frame_id"] + cell_headers)

        for key, values in mean_attention_map.items():
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


def get_total_cells(event_config) -> int:
    total_cells = 0
    for grid in event_config['grids']:
        cells = grid["rows"] * grid["cols"]
        total_cells += cells
    return total_cells


def save_cell_value_subplots(mean_attention_map, output_path, display_results, total_cells):
    plot_cols = 6
    plot_rows = math.ceil(total_cells / plot_cols)

    fig, axs = plt.subplots(
        nrows=plot_rows,
        ncols=plot_cols,
        figsize=(plot_cols * 4, plot_rows * 3),
    )

    # Flatten the axes array for easy iteration, in case of a single row/column
    axs = np.atleast_2d(axs).reshape(-1)

    for cell_index in range(total_cells):
        ax = axs[cell_index]
        ax.plot(
            [value[cell_index] for value in mean_attention_map.values()],
            label=f"Cell {cell_index+1}",
        )
        ax.set_title(f"Cell {cell_index+1}")

        if cell_index == 0:
            ax.legend()

    # Hide unused subplots if any
    for ax in axs[total_cells:]:
        ax.axis('off')

    fig.text(0.5, 0.04, "Frame", ha="center")
    fig.text(0.04, 0.5, "Mean Attention", va="center", rotation="vertical")

    plt.subplots_adjust(
        left=0.07, bottom=0.1, right=0.97, top=0.95, wspace=0.2, hspace=0.4
    )

    plt.savefig(os.path.join(output_path, "cell_value_subplots.png"))

    if display_results:
        plt.show()
    else:
        plt.close()


def save_combined_plot(mean_attention_map, output_path, display_results, total_cells):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Generate a color cycle or define a list of colors if specific ones are desired
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for cell_index in range(total_cells):
        ax.plot(
            [value[cell_index] for value in mean_attention_map.values()],
            label=f"Cell {cell_index + 1}",
            color=color_cycle[cell_index % len(color_cycle)]
        )

    ax.set_title("Cell Values Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean Attention")
    ax.legend()

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    plt.savefig(os.path.join(output_path, "combined_cell_values_plot.png"))

    if display_results:
        plt.show()
    else:
        plt.close()


def process_video_and_generate_attention_map(
    heatmap_video_path,
    original_video_path,
    target_path,
    video_id,
    data_config,
    event_config,
) -> dict:
    ensure_matching_video_resolution(original_video_path, heatmap_video_path)

    cap_heatmap = cv2.VideoCapture(heatmap_video_path)
    cap_original = cv2.VideoCapture(original_video_path)

    ret_heatmap, frame_heatmap = cap_heatmap.read()
    ret_original, frame_original = cap_original.read()

    total_frames_heatmap = int(cap_heatmap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_original = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames_heatmap != total_frames_original:
        print("Number of frames in input videos do not match.")
        print(f"Heatmap video: {total_frames_heatmap} frames")
        print(f"Original video: {total_frames_original} frames")
        exit(1)

    if not ret_heatmap and ret_original:
        print("Unable to read video(s).")
        exit(1)

    out = cv2.VideoWriter(
        os.path.join(target_path, f"{video_id}_attention_grid.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        data_config["fps"],
        (frame_original.shape[1], frame_original.shape[0]),
    )

    cell_positions = calculate_grid_cell_positions(frame_heatmap, event_config)
    mean_attention_map = {}

    frame_number = 0
    with tqdm(total=total_frames_heatmap, desc="Frame progress", leave=False) as pbar_frames:
        while ret_heatmap and ret_original:
            frame_number += 1

            mean_attention_map[frame_number] = calculate_cell_values(
                frame_heatmap, cell_positions, event_config["saliency_threshold"]
            )
            combined_frame = overlay_heatmap(frame_heatmap, frame_original)

            draw_grid(combined_frame, cell_positions)

            annotate_frame(
                combined_frame,
                f"frame: {frame_number}. saliency_threshold: {event_config['saliency_threshold']}",
                (10, 30),
            )

            out.write(combined_frame)

            if args.display_results:
                cv2.imshow("Saliency grid", combined_frame)

            ret_heatmap, frame_heatmap = cap_heatmap.read()
            ret_original, frame_original = cap_original.read()

            if cv2.waitKey(30) & 0xFF == ord("q"):
                print("Interrupted by user")
                break

            pbar_frames.update(1)

    cap_heatmap.release()
    cap_original.release()
    out.release()
    cv2.destroyAllWindows()

    return mean_attention_map


def main(data_path, output_path, data_config, event_config, display_results):
    assert os.path.exists(output_path), f"Output path {output_path} does not exist."

    total_cells = get_total_cells(event_config)

    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(output_path, video_id)

        if not os.path.isdir(target_path):
            tqdm_obj.write(f"Skipping {video_id}: Output directory does not exist")
            continue

        original_video_path = os.path.join(video_path, f"{video_id}.avi")
        heatmap_video_path = os.path.join(target_path, f"{video_id}_heatmap.avi")

        if not os.path.exists(heatmap_video_path) or not os.path.exists(original_video_path):
            tqdm_obj.write(f"Skipping {video_id}: Heatmap or original videos do not exist")
            continue

        mean_attention_map = process_video_and_generate_attention_map(
            heatmap_video_path,
            original_video_path,
            target_path,
            video_id,
            data_config,
            event_config,
        )

        save_csv(mean_attention_map, target_path, event_config)
        save_cell_value_subplots(mean_attention_map, target_path, display_results, total_cells)
        save_combined_plot(mean_attention_map, target_path, display_results, total_cells)

    save_config(output_path, data_config, event_config)

    print("grid_attention.py completed.")


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
