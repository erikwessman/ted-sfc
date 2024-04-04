import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import cv2
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import helper


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the directory containing the output for each video, including video and cell values to visualize",
    )

    return parser.parse_args()


def visualize_cell_values(video_path, cell_values, output_path):
    total_cells = cell_values.shape[1] - 1
    number_of_frames = cell_values.shape[0]

    plot_cols = 6
    plot_rows = math.ceil(total_cells / plot_cols)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dpi = 100

    # Calculate the height of the subplot, with a margin of 100 pixels
    subplot_height = int(plot_rows * frame_height / plot_cols) + 100

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height + subplot_height))

    frame_number = 0
    with tqdm(total=number_of_frames, desc="Frame progress", leave=False) as pbar_frames:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(frame_width / dpi, subplot_height / dpi),
                                    gridspec_kw={'width_ratios': [1] * plot_cols, 'left': 0.08, 'right': 0.95, 'top': 0.8, 'bottom': 0.22}, dpi=dpi, sharex=True, sharey=True)
            fig.text(0.5, 0.05, 'Frame Number', ha='center', va='center')
            fig.text(0.02, 0.5, 'Mean Cell Value', ha='center', va='center', rotation='vertical')
            canvas = FigureCanvas(fig)

            for cell_index in range(total_cells):
                ax = axs.flat[cell_index]
                ax.set_xlim(0, number_of_frames)
                ax.set_ylim(0, 1)
                ax.plot(cell_values.iloc[:frame_number, cell_index + 1])
                ax.set_title(f"Cell {cell_index + 1}")

            for ax in axs.flat[total_cells:]:
                ax.set_visible(False)

            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            plot_img_rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

            plot_img_resized = cv2.resize(plot_img_rgb, (frame_width, subplot_height))

            combined_image = np.vstack((frame, plot_img_resized))
            out.write(combined_image)
            plt.close(fig)

            frame_number += 1
            pbar_frames.update(1)

        cap.release()
        out.release()


def main(data_path: str, args):
    for _, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if not os.path.isfile(os.path.join(target_path, "cell_values.csv")):
            tqdm_obj.write(f"Skipped. File 'cell_values.csv' not found in {target_path}.")
            continue

        cell_values = pd.read_csv(os.path.join(target_path, "cell_values.csv"), sep=";")

        grid_video_path = glob.glob(os.path.join(target_path, "*_grid.avi"))

        if len(grid_video_path) > 1:
            tqdm_obj.write(f"Skipped. Multiple grid videos found in {target_path}.")
            continue
        elif len(grid_video_path) == 0:
            tqdm_obj.write(f"Skipped. No source grid video found in {target_path}.")
            continue
        else:
            output_path = os.path.join(target_path, f"{video_id}_plot_viz.avi")
            visualize_cell_values(grid_video_path[0], cell_values, output_path)

    print("visualize.py completed.")


if __name__ == "__main__":
    args = parse_arguments()

    main(
        args.data_path,
        args
    )
