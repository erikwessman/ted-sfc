import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Constants and configurations
HEATMAP_VIDEO_PATH = "./data/heatmap.avi"
ORIGINAL_VIDEO_PATH = "./data/original.webm"
OUTPUT_PATH = "./output/attention"
THRESHOLD = 50
TOP_LEFT = (0, 0.4)
BOTTOM_RIGHT = (0.6, 0.6)
NUM_COLS = 6
NUM_ROWS = 2
COMMON_COLOR = (255, 255, 255)


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


def draw_grid(frame, cell_positions, line_color=(0, 255, 0), line_thickness=1):
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


def calculate_cell_positions(image):
    h, w, _ = image.shape  # Get the height and width of the frame

    # Convert proportional positions to pixel positions
    start_x = int(w * TOP_LEFT[0])
    start_y = int(h * TOP_LEFT[1])
    end_x = int(w * BOTTOM_RIGHT[0])
    end_y = int(h * BOTTOM_RIGHT[1])

    # Calculate the size of each cell in the grid
    cell_width = (end_x - start_x) // NUM_COLS
    cell_height = (end_y - start_y) // NUM_ROWS

    cell_positions = []

    # Iterate over the cells
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
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

        # Get the value if its over the threshold, otherwise 0
        # cell_mean_value = (cell_mean_value if cell_mean_value >= THRESHOLD else 0)

        heatmap_mean_values.append(cell_mean_value)

        # Draw a green bounding box around the cell if the value is above the threshold
        if cell_mean_value > THRESHOLD:
            cv2.rectangle(
                frame,
                (cell_start_x, cell_start_y),
                (cell_end_x, cell_end_y),
                (0, 0, 255),
                2,
            )

    return heatmap_mean_values


def overlay_heatmap(frame_heatmap, frame_original):
    heatmap = cv2.applyColorMap(
        (frame_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(frame_original, 0.5, heatmap, 0.5, 0)


def annotate_frame(
    frame, text, position, font_scale=1, font_color=(255, 255, 255), thickness=2
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)


def save_results(angle_differences_per_frame):
    csv_path = os.path.join(OUTPUT_PATH, "output.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame"] + [f"Cell{i+1}" for i in range(12)])
        for frame_data in angle_differences_per_frame:
            writer.writerow(frame_data)


def plot_results(angle_differences):
    raise NotImplementedError()

    for cell_index, diffs in angle_differences.items():
        plt.figure()
        plt.plot(diffs, label=f"Cell {cell_index+1}")
        plt.xlabel("Frame")
        plt.ylabel("Angle Difference")
        plt.title(f"Angle Difference for Cell {cell_index+1} Over Time")
        plt.legend()
    plt.show()


def main():
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
        os.path.join(OUTPUT_PATH, "output.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        20.0,
        (frame_original.shape[1], frame_original.shape[0]),
    )

    frame_number = 0
    mean_attention_map = {}
    cell_positions = calculate_cell_positions(frame_heatmap)

    while ret_heatmap and ret_original:
        frame_number += 1

        # Calculate the mean heatmap value for each cell
        mean_attention_map[frame_number] = process_frame(frame_heatmap, cell_positions)

        # Overlay cell grid on heatmap frame
        draw_grid(frame_heatmap, cell_positions)

        # Annotate frame with the frame number
        annotate_frame(frame_heatmap, f"Frame: {frame_number}", (10, 30))

        # Overlay the heatmap frame over the original frame
        combined_frame = overlay_heatmap(frame_heatmap, frame_original)

        # Write frame to video output
        # out.write(combined_frame)

        # Show the frame to the user
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

    save_results(mean_attention_map)
    plot_results(mean_attention_map)

    print("Done")


if __name__ == "__main__":
    main()
