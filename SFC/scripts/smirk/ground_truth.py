import argparse
import pandas as pd
import yaml


# Camera constants
SENSOR_WIDTH = 3.13  # cm
SENSOR_HEIGHT = 2.00  # cm
FOCAL_LENGTH = 3.73  # cm
IMAGE_WIDTH = 752
IMAGE_HEIGHT = 480
ANGLE_OF_VIEW = 45  # degrees

# Pixel size calculation
PIXEL_SIZE_WIDTH = SENSOR_WIDTH / IMAGE_WIDTH
PIXEL_SIZE_HEIGHT = SENSOR_HEIGHT / IMAGE_HEIGHT

# Heuristically determined positions
ROAD_X_MIN = -220
ROAD_X_MAX = 220


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process SMIRK dataset into videos.")
    parser.add_argument(
        "labels_path",
        type=str,
        help="Path to labels.csv",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output path",
    )
    return parser.parse_args()


def pedestrian_position(distance, bbox_center):
    """
    Calculate the real-world position of a pedestrian.

    Args:
        distance (float): The distance of the pedestrian from the camera (in cm).
        bbox_center (tuple): The center (x, y) of the pedestrian's bounding box in image coordinates.

    Returns:
        A tuple representing the (X, Y, Z) coordinates in the real world, where
        X is the horizontal displacement,
        Y is the vertical displacement,
        Z is the distance from the camera.
    """
    # Convert image coordinates to sensor coordinates
    sensor_x = (bbox_center[0] - (IMAGE_WIDTH / 2)) * PIXEL_SIZE_WIDTH
    # Inverting Y to match real-world coordinates
    sensor_y = ((IMAGE_HEIGHT / 2) - bbox_center[1]) * PIXEL_SIZE_HEIGHT

    # Calculate real-world X, Y based on similar triangles
    real_world_x = (sensor_x / FOCAL_LENGTH) * distance
    real_world_y = (sensor_y / FOCAL_LENGTH) * distance

    # The real-world Z coordinate is the distance from the camera
    real_world_z = distance

    return (real_world_x, real_world_y, real_world_z)


def get_bbox_center(x_min, x_max, y_min, y_max):
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y


def save_yml(path: str, data):
    with open(path, 'w', encoding="utf-8") as yaml_file:
        dump = yaml.safe_dump(data,
                              default_flow_style=False,
                              allow_unicode=False,
                              encoding=None,
                              sort_keys=False,
                              line_break=10)
        yaml_file.write(dump)


def main(labels_path: str, output_path: str):
    """
    Find the ground truth for a set of SMIRK videos
    """
    df = pd.read_csv(labels_path)

    # Only left-to-right crossings
    filtered_df = df[df["scenario_type"] == "left"]

    # TODO: only get those that start and end a reasonable distance from the camera
    # TODO: maybe get events for both directions

    # Map video IDs to ground truths
    video_map = {}

    for _, row in filtered_df.iterrows():
        video_id = row["run_id"]
        frame_index = row["frame_index"]

        x_min = row["x_min"]
        x_max = row["x_max"]
        y_min = row["y_min"]
        y_max = row["y_max"]

        bbox_center = get_bbox_center(x_min, x_max, y_min, y_max)
        distance_from_camera = row["current_distance"] * 100  # convert m to cm

        position = pedestrian_position(distance_from_camera, bbox_center)
        x, _, _ = position

        # The pedestian is inside the road
        if x and ROAD_X_MIN <= x <= ROAD_X_MAX:
            if video_id in video_map:
                video_map[video_id].append(frame_index)
            else:
                video_map[video_id] = [frame_index]

    annotations = []

    # Inefficient, this could be done in the previous loop but i dont care
    for video_id, frames in video_map.items():
        annotation_map = {
            "id": video_id,
            "event_window": [min(frames), max(frames)],
            "direction": "right"
        }
        annotations.append(annotation_map)

    # save_yml(output_path, annotations)

    print(annotations)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.labels_path, args.output_path)
