import os
import csv
import cv2
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_config(file_path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def match_video_resolutions(original_video_path: str, video_to_resize_path: str) -> str:
    # Open the original video and get its resolution
    original_cap = cv2.VideoCapture(original_video_path)
    original_width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_cap.release()

    # Open the new video and get its resolution
    new_cap = cv2.VideoCapture(video_to_resize_path)
    new_width = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if the resolutions are the same
    if original_width == new_width and original_height == new_height:
        new_cap.release()
        print("Videos already have the same resolution. No resizing needed.")
        return video_to_resize_path
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = new_cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(
            video_to_resize_path, fourcc, fps, (original_width, original_height)
        )

        # Read through the new video, resize each frame, and write to the output
        while new_cap.isOpened():
            ret, frame = new_cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (original_width, original_height))
            out.write(resized_frame)

        new_cap.release()
        out.release()
        print(f"New video has been resized and saved to {video_to_resize_path}.")
        return video_to_resize_path


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
        cell_mean_value = (
            cell_mean_value if cell_mean_value >= saliency_threshold else 0
        )

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


def save_csv(mean_attention_map, output_path):
    csv_path = os.path.join(output_path, "cell_values.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["frame_id"] + [f"cell{i+1}" for i in range(6)])
        for key, values in mean_attention_map.items():
            row = [key] + values[0:6]
            writer.writerow(row)


def save_config(output_path, args, config):
    config_path = os.path.join(output_path, "config.txt")
    with open(config_path, "w") as f:
        for key, value in config.items():
            line = f"{key}: {value}\n"
            f.write(line)
    print(f"Config saved to {config_path}")


def save_plots(mean_attention_map, output_path, display_results, config):
    fig, axs = plt.subplots(
        nrows=config["grid_num_rows"],
        ncols=config["grid_num_cols"],
        figsize=(config["grid_num_cols"] * 4, config["grid_num_rows"] * 3),
    )

    for cell_index, ax in enumerate(axs.flat):
        ax.plot(
            [value[cell_index] for value in mean_attention_map.values()],
            label=f"Cell {cell_index+1}",
        )
        ax.set_title(f"Cell {cell_index+1}")

        if cell_index == 0:
            ax.legend()

    fig.text(0.5, 0.04, "Frame", ha="center")
    fig.text(0.04, 0.5, "Mean Attention", va="center", rotation="vertical")

    plt.subplots_adjust(
        left=0.07, bottom=0.1, right=0.97, top=0.95, wspace=0.2, hspace=0.4
    )

    plt.savefig(os.path.join(output_path, "all_cells.png"))

    if display_results:
        plt.show()


def process_video_and_generate_attention_map(
    heatmap_video_path, original_video_path, target_path, video_id, config
) -> dict:
    # Resize the heatmap video to have the same dimensions as the original video
    match_video_resolutions(heatmap_video_path, original_video_path)

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
        print("Unable to read video(s)")
        exit(1)

    out = cv2.VideoWriter(
        os.path.join(target_path, f"{video_id}_attention_grid.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        20.0,
        (frame_original.shape[1], frame_original.shape[0]),
    )

    frame_number = 0
    mean_attention_map = {}

    cell_positions = calculate_cell_positions(frame_heatmap, config)

    with tqdm(total=total_frames_heatmap, desc="Processing frames") as progress:
        while ret_heatmap and ret_original:
            frame_number += 1

            mean_attention_map[frame_number] = calculate_cell_values(
                frame_heatmap, cell_positions, config["saliency_threshold"]
            )
            combined_frame = overlay_heatmap(frame_heatmap, frame_original)

            draw_grid(combined_frame, cell_positions)

            annotate_frame(
                combined_frame,
                f"frame: {frame_number}. saliency_threshold: {config['saliency_threshold']}",
                (10, 30),
            )

            out.write(combined_frame)

            if args.display_results:
                cv2.imshow("Saliency grid", combined_frame)

            ret_heatmap, frame_heatmap = cap_heatmap.read()
            ret_original, frame_original = cap_original.read()

            progress.update(1)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                print("Interrupted by user")
                break

    cap_heatmap.release()
    cap_original.release()
    out.release()
    cv2.destroyAllWindows()

    return mean_attention_map


def main(args):
    config = load_config(args.grid_config_path)

    dataset_path = args.dataset_path
    output_path = args.output_path

    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."
    assert os.path.exists(output_path), f"Output path {output_path} does not exist."

    video_dirs = [
        name
        for name in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, name))
    ]

    for video_id in video_dirs:
        video_path = os.path.join(dataset_path, video_id)
        target_path = os.path.join(output_path, video_id)

        print(f"Processing folder {video_id}")
        if os.path.isdir(video_path) and os.path.isdir(target_path):
            original_video_path = os.path.join(video_path, f"{video_id}.avi")
            heatmap_video_path = os.path.join(target_path, f"{video_id}_heatmap.avi")

            if os.path.exists(heatmap_video_path) and os.path.exists(
                original_video_path
            ):
                mean_attention_map = process_video_and_generate_attention_map(
                    heatmap_video_path,
                    original_video_path,
                    target_path,
                    video_id,
                    config,
                )

                save_csv(mean_attention_map, target_path)
                save_plots(
                    mean_attention_map, target_path, args.display_results, config
                )
                print(f"Done. Results saved in {target_path}.")
            else:
                print("Skipped. Source or heatmap video does not exist.")
        else:
            print("Skipped. Source or output video directory does not exist.")

        print("--------------------------")

    save_config(output_path, args, config)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "dataset_path",
        help="Path to the folder containing the dataset, e.g. ./data/{dataset_name}",
    )
    parser.add_argument(
        "output_path",
        help="Path to the folder where the output will be saved, e.g. ./output/{dataset_name}",
    )
    parser.add_argument(
        "grid_config_path",
        help="Path to config yml file",
    )
    parser.add_argument("--display-results", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(args)
