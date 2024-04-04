import os
import yaml
import csv
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def traverse_videos(data_path: str):
    """
    Given a data path, traverses the directory and yields the video path, video ID and the tqdm object.
    Optionally shows a progress bar.

    :param data_path: Path to the directory containing video folders.
    :param show_progress: If True, shows a progress bar. Defaults to True.
    """
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"

    video_dirs = [
        name
        for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    ]

    video_dirs = tqdm(video_dirs, desc="Processing folders")

    for video_id in video_dirs:
        video_dirs.set_description(f"Processing folder {video_id}")

        video_dir = os.path.join(data_path, video_id)

        yield video_dir, video_id, tqdm

        video_dirs.set_description("Processing folders")


def load_yml(file_path: str) -> dict:
    """
    Load a YML file as a dict
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_ground_truth(ground_truth: dict, video_id: str):
    """
    Gets the ground truth associated with a video ID
    """
    for video in ground_truth:
        if video["id"] == video_id:
            return video
    return None


def save_cell_value_csv(cell_value_map, output_path, grids_config):
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

        for key, values in cell_value_map.items():
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

def save_cell_value_subplots(cell_value_map, output_path, display_results, y_label):
    total_cells = len(list(cell_value_map.values())[0])

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
            [value[cell_index] for value in cell_value_map.values()],
            label=f"Cell {cell_index+1}",
        )
        ax.set_title(f"Cell {cell_index+1}")

        if cell_index == 0:
            ax.legend()

    # Hide unused subplots if any
    for ax in axs[total_cells:]:
        ax.axis('off')

    fig.text(0.5, 0.04, "Frame", ha="center")
    fig.text(0.04, 0.5, y_label, va="center", rotation="vertical")

    plt.subplots_adjust(
        left=0.07, bottom=0.1, right=0.97, top=0.95, wspace=0.2, hspace=0.4
    )

    plt.savefig(os.path.join(output_path, "cell_value_subplots.png"))

    if display_results:
        plt.show()
    else:
        plt.close()


def save_combined_plot(cell_value_map, output_path, display_results, y_label):
    fig, ax = plt.subplots(figsize=(10, 7))
    total_cells = len(list(cell_value_map.values())[0])

    # Generate a color cycle or define a list of colors if specific ones are desired
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for cell_index in range(total_cells):
        ax.plot(
            [value[cell_index] for value in cell_value_map.values()],
            label=f"Cell {cell_index + 1}",
            color=color_cycle[cell_index % len(color_cycle)]
        )

    ax.set_title("Cell values over time")
    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    ax.legend()

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    plt.savefig(os.path.join(output_path, "combined_cell_values_plot.png"))

    if display_results:
        plt.show()
    else:
        plt.close()


def get_total_cells(event_config) -> int:
    total_cells = 0
    for grid in event_config['grids']:
        cells = grid["rows"] * grid["cols"]
        total_cells += cells
    return total_cells

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

    for grid_config in grid_config["grids"]:
        # Convert proportional positions to pixel positions for each grid
        start_x = int(w * grid_config["top_left"][0])
        start_y = int(h * grid_config["top_left"][1])
        end_x = int(w * grid_config["bottom_right"][0])
        end_y = int(h * grid_config["bottom_right"][1])

        cell_width = (end_x - start_x) // grid_config["cols"]
        cell_height = (end_y - start_y) // grid_config["rows"]

        cell_positions = []

        for row in range(grid_config["rows"]):
            for col in range(grid_config["cols"]):
                cell_start_x = start_x + col * cell_width
                cell_start_y = start_y + row * cell_height

                cell_end_x = cell_start_x + cell_width
                cell_end_y = cell_start_y + cell_height

                cell_positions.append(
                    ((cell_start_x, cell_start_y), (cell_end_x, cell_end_y))
                )

        all_cell_positions.extend(cell_positions)

    return all_cell_positions

def annotate_frame(
    frame, text, position, font_scale=1, font_color=(255, 255, 255), thickness=2
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)
