import os
import yaml
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


def get_ground_truth(ground_truth: dict, video_id: str, direction: str):
    """
    Gets the ground truth associated with a video ID and an event direction
    """
    for video in ground_truth:
        if video["id"] == video_id and video["direction"] == direction:
            return video
    return None
