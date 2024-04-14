import argparse
import pandas as pd
import yaml
import random

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

# Heuristically determined positions (cm)
ROAD_X_MIN = -220
ROAD_X_MAX = 220


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process SMIRK dataset into videos.")
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


def get_pedestrian_position(distance, bbox_center):
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
    with open(path, "a", encoding="utf-8") as yaml_file:
        dump = yaml.safe_dump(
            data,
            default_flow_style=False,
            allow_unicode=False,
            encoding=None,
            sort_keys=False,
            line_break=10,
        )
        yaml_file.write(dump)


def get_df_ground_truth(df):
    """
    Get the ground truth from a SMIRK labels dataframe
    """
    video_id = None
    event_start = 0
    event_end = 0
    scenario_type = None

    for _, row in df.iterrows():
        video_id = row["run_id"]
        scenario_type = row["scenario_type"]
        frame_index = row["frame_index"]

        x_min = row["x_min"]
        x_max = row["x_max"]
        y_min = row["y_min"]
        y_max = row["y_max"]

        bbox_center = get_bbox_center(x_min, x_max, y_min, y_max)
        distance_from_camera = row["current_distance"] * 100  # convert m to cm

        x, _, _ = get_pedestrian_position(distance_from_camera, bbox_center)

        # The pedestian is inside the road
        if x and ROAD_X_MIN <= x <= ROAD_X_MAX:
            if not event_start:
                event_start = frame_index

            if not event_end or event_end < frame_index:
                event_end = frame_index

    return {
        "id": video_id,
        "event_window": [event_start, event_end],
        "type": scenario_type,
    }


def main(labels_path: str, output_path: str):
    """
    Find the ground truth for a set of SMIRK videos
    """
    df = pd.read_csv(labels_path)
    group = df.groupby("run_id")

    annotations = []

    for group_id, df_group in group:
        if not (df_group["scenario_type"].isin(["left", "right"])).any():
            # Skip scenarios that don't contain pedestrians crossing the road
            continue

        if (df_group["current_distance"] > 26).any():
            # Skip scenarios in which the pedestrian is further than 26m away
            continue

        if (df_group["class_text"] == "background").all():
            # Skip scenario if all values in 'class_text' are 'background'
            continue

        gt = get_df_ground_truth(df_group)
        annotations.append(gt)

    print(f"Found {len(annotations)} valid crossings in {len(group)} videos")

    save_yml(output_path, annotations)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.labels_path, args.output_path)
