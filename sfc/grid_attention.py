import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Constants and configurations
HEATMAP_VIDEO_PATH = "./data/heatmap.avi"
ORIGINAL_VIDEO_PATH = "./data/original.webm"
OUTPUT_PATH = "./output/attention"
SALIENCY_THRESHOLD = 50
GRID_TOP_LEFT = (0, 0.35)
GRID_BOTTOM_RIGHT = (0.5, 0.6)
GRID_NUM_COLS = 6
GRID_NUM_ROWS = 2


def match_video_resolutions(new_video_path: str, original_video_path: str):
    # Open the original video and get its resolution
    original_cap = cv2.VideoCapture(original_video_path)
    original_width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_cap.release()

    # Open the new video and get its resolution
    new_cap = cv2.VideoCapture(new_video_path)
    new_width = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if the resolutions are the same
    if original_width == new_width and original_height == new_height:
        new_cap.release()
        print("Videos already have the same resolution. No resizing needed.")
        return new_video_path
    else:
        resized_new_video_path = os.path.join(OUTPUT_PATH, "resized.avi")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = new_cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(
            resized_new_video_path, fourcc, fps, (original_width, original_height)
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
        print(f"New video has been resized and saved to {resized_new_video_path}.")
        return resized_new_video_path


def draw_grid(frame, cell_positions, line_color=(255, 255, 255), line_thickness=1):
    for top_left, bottom_right in cell_positions:
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
        cell_index = cell_positions.index((top_left, bottom_right))
        cv2.putText(
            frame,
            str(cell_index + 1),
            (start_x + 20, start_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )


def calculate_cell_positions(image, top_left, bottom_right, num_cols, num_rows):
    h, w, _ = image.shape  # Get the height and width of the frame

    # Convert proportional positions to pixel positions
    start_x = int(w * top_left[0])
    start_y = int(h * top_left[1])
    end_x = int(w * bottom_right[0])
    end_y = int(h * bottom_right[1])

    # Calculate the size of each cell in the grid
    cell_width = (end_x - start_x) // num_cols
    cell_height = (end_y - start_y) // num_rows

    cell_positions = []

    # Iterate over the cells
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the coordinates of the top-left corner of the cell
            cell_start_x = start_x + col * cell_width
            cell_start_y = start_y + row * cell_height

            # Calculate the coordinates of the bottom-right corner of the cell
            cell_end_x = cell_start_x + cell_width
            cell_end_y = cell_start_y + cell_height

            # Append the cell coordinates to the list
            cell_positions.append(
                ((cell_start_x, cell_start_y), (cell_end_x, cell_end_y))
            )

    return cell_positions


def process_frame(frame, cell_positions):
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
        cell_mean_value = np.mean(cell_region)

        # Get the value if its over the SALIENCY_THRESHOLD, otherwise 0
        # cell_mean_value = (cell_mean_value if cell_mean_value >= SALIENCY_THRESHOLD else 0)

        heatmap_mean_values.append(cell_mean_value)

        # Draw a bounding box around the cell if the value is above the SALIENCY_THRESHOLD
        if cell_mean_value > SALIENCY_THRESHOLD:
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


def save_results(mean_attention_map, output_path):
    csv_path = os.path.join(output_path, "cell_values.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["frame_id"] + [f"cell{i+1}" for i in range(6)])
        for key, values in mean_attention_map.items():
            row = [key] + values[0:6]
            writer.writerow(row)


def create_plots(mean_attention_map, output_path, plot_results):
    for cell_index in range(12):
        plt.figure()
        plt.plot(
            [value[cell_index] for value in mean_attention_map.values()],
            label=f"Cell {cell_index+1}",
        )
        plt.xlabel("Frame")
        plt.ylabel("Mean attention")
        plt.title(f"Mean attention value for cell {cell_index+1} over time")
        plt.legend()
        plt.savefig(os.path.join(output_path, f"cell_{cell_index+1}.png"))

    if (plot_results):
        plt.show()


def main(args):
    if not os.path.exists(OUTPUT_PATH):
        print("Output path does not exist, creating it...")
        os.makedirs(OUTPUT_PATH)

    output_path = os.path.join(
        OUTPUT_PATH, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(output_path)

    # Resize the heatmap video to have the same dimensions as the original video
    heatmap_video_path = match_video_resolutions(
        HEATMAP_VIDEO_PATH, ORIGINAL_VIDEO_PATH
    )

    cap_heatmap = cv2.VideoCapture(heatmap_video_path)
    cap_original = cv2.VideoCapture(ORIGINAL_VIDEO_PATH)

    ret_heatmap, frame_heatmap = cap_heatmap.read()
    ret_original, frame_original = cap_original.read()

    if not ret_heatmap and ret_original:
        print("Unable to read video")
        return

    out = cv2.VideoWriter(
        os.path.join(output_path, "attention_grid.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        20.0,
        (frame_original.shape[1], frame_original.shape[0]),
    )

    frame_number = 0
    mean_attention_map = {}
    cell_positions = calculate_cell_positions(
        frame_heatmap, GRID_TOP_LEFT, GRID_BOTTOM_RIGHT, GRID_NUM_COLS, GRID_NUM_ROWS
    )

    while ret_heatmap and ret_original:
        frame_number += 1

        # Calculate the mean heatmap value for each cell
        mean_attention_map[frame_number] = process_frame(frame_heatmap, cell_positions)

        # Overlay the heatmap frame over the original frame
        combined_frame = overlay_heatmap(frame_heatmap, frame_original)

        # Overlay cell grid on combined frame
        draw_grid(combined_frame, cell_positions)

        # Add frame number
        annotate_frame(combined_frame, f"Frame: {frame_number}", (10, 30))

        # Write frame to video output
        out.write(combined_frame)

        # Show the frame to the user
        if (args.plot_results):
            cv2.imshow("Saliency grid", combined_frame)

        ret_heatmap, frame_heatmap = cap_heatmap.read()
        ret_original, frame_original = cap_original.read()

        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("Interrupted by user")
            break

    cap_heatmap.release()
    cap_original.release()
    out.release()
    cv2.destroyAllWindows()

    save_results(mean_attention_map, output_path)
    create_plots(mean_attention_map, output_path, args.plot_results)

    print(f"Saved results to {output_path}")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
            "--plot-results", action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()

    main(args)
