import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

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
        "config_path",
        help="Path to the config yml file",
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


def keep_largest(values):
    """Keep the largest value in the array, while making all other values 0"""
    max_index = values.index(max(values))
    return [values[i] if i == max_index else 0 for i in range(len(values))]


def calculate_cell_values(frame, cell_positions, saliency_threshold):
    h, w = frame.shape[:2]
    heatmap_mean_values = []
    max_value = 0
    max_index = -1  # Initialize with an invalid index

    # Calculate all mean values and find the maximum
    for index, (top_left, bottom_right) in enumerate(cell_positions):
        cell_start_x, cell_start_y = top_left
        cell_end_x, cell_end_y = min(bottom_right[0], w), min(bottom_right[1], h)
        cell_region = frame[cell_start_y:cell_end_y, cell_start_x:cell_end_x]
        normalized_cell_mean_value = np.mean(cell_region) / 255

        heatmap_mean_values.append(normalized_cell_mean_value)

        # Update the maximum value and index if the current value is larger
        if normalized_cell_mean_value > max_value:
            max_value = normalized_cell_mean_value
            max_index = index

    # Set all values to 0 except the maximum value
    result_values = []
    for index, value in enumerate(heatmap_mean_values):
        if index == max_index and value >= saliency_threshold:
            result_values.append(value)
        else:
            result_values.append(0)

    # Draw a rectangle around the cell with the maximum value
    if max_index != -1 and heatmap_mean_values[max_index] >= saliency_threshold:
        top_left = cell_positions[max_index][0]
        bottom_right = cell_positions[max_index][1]
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 5)

    return result_values


def overlay_heatmap(frame_heatmap, frame_original):
    heatmap = cv2.applyColorMap(
        (frame_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(frame_original, 0.6, heatmap, 0.4, 0)


def process_video_and_generate_attention_map(
    heatmap_video_path,
    original_video_path,
    target_path,
    video_id,
    config,
    display_results,
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
        config["fps"],
        (frame_original.shape[1], frame_original.shape[0]),
    )

    threshold = config["attention"]["saliency_threshold"]

    cell_positions = helper.calculate_grid_cell_positions(frame_heatmap, config)
    mean_attention_map = {}

    frame_number = 0
    with tqdm(
        total=total_frames_heatmap, desc="Frame progress", leave=False
    ) as pbar_frames:
        while ret_heatmap and ret_original:
            frame_number += 1

            mean_attention_map[frame_number] = calculate_cell_values(
                frame_heatmap, cell_positions, threshold
            )
            combined_frame = overlay_heatmap(frame_heatmap, frame_original)

            helper.draw_grid(combined_frame, cell_positions)

            helper.annotate_frame(
                combined_frame,
                f"frame: {frame_number}. saliency_threshold: {threshold}",
                (10, 30),
            )

            out.write(combined_frame)

            if display_results:
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


def main(
    data_path: str, output_path: str, config_path: str, display_results: bool = False
):
    # Load config
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]

    assert os.path.exists(output_path), f"Output path {output_path} does not exist."

    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(output_path, video_id)

        if not os.path.isdir(target_path):
            tqdm_obj.write(f"Skipping {video_id}: Output directory does not exist")
            continue

        original_video_path = os.path.join(video_path, f"{video_id}.avi")
        heatmap_video_path = os.path.join(target_path, f"{video_id}_heatmap.avi")

        if not os.path.exists(heatmap_video_path) or not os.path.exists(
            original_video_path
        ):
            tqdm_obj.write(
                f"Skipping {video_id}: Heatmap or original videos do not exist"
            )
            continue

        mean_attention_map = process_video_and_generate_attention_map(
            heatmap_video_path,
            original_video_path,
            target_path,
            video_id,
            grid_config,
            display_results,
        )

        helper.save_cell_value_csv(mean_attention_map, target_path, grid_config)

        plot_path = os.path.join(target_path, "plots")
        os.makedirs(plot_path, exist_ok=True)

        helper.save_cell_value_subplots(
            mean_attention_map, plot_path, display_results, "Mean attention"
        )
        helper.save_combined_plot(
            mean_attention_map, plot_path, display_results, "Mean attention"
        )

    helper.save_config(config, output_path, "config.yml")


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.data_path,
        args.output_path,
        args.config_path,
        args.display_results,
    )
