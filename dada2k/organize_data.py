import os
import argparse
import scipy.io
import numpy as np
import ffmpeg
import subprocess


def main(input_path: str, output_path: str):
    rgb_videos_path = os.path.join(input_path, "rgb_videos")
    salmap_videos_path = os.path.join(input_path, "salmap_videos")
    coordinate_path = os.path.join(input_path, "coordinate")
    
    if not os.path.exists(input_path):
        print("Path does not exist")
        exit(1)

    for category in os.listdir(input_path):
        category_path = os.path.join(input_path, category)
        if not os.path.isdir(category_path) or category in ["coordinate", "rgb_videos", "salmap_videos"]:
            continue

        for video in os.listdir(category_path):
            video_path = os.path.join(category_path, video)
            if not os.path.isdir(video_path):
                continue
            
            # get all images, move to rgb_videos/category/video
            tmp_rgb_path = os.path.join(rgb_videos_path, category, video)
            os.makedirs(tmp_rgb_path, exist_ok=True)
            rename_command = f"find '{os.path.join(video_path, 'images')}' -name '*.png' -exec sh -c 'mv \"$1\" \"${{1%.png}}.jpg\"' _ {{}} \;"

            # Execute the command
            subprocess.run(rename_command, shell=True, check=True)
            subprocess.run(f"ffmpeg -framerate 24 -i {os.path.join(video_path, 'images/%4d.jpg')} -c:v libx264 -pix_fmt yuv420p {os.path.join(rgb_videos_path, category, f'{video}.avi')}", shell=True, check=True)
            # ffmpeg.input(os.path.join(video_path, "images", "%4d.jpg"), pattern_type='glob').output(os.path.join(rgb_videos_path, category, f"{video}.avi")).run()
    
            # shutil.move(os.path.join(video_path, "images", "*"), os.path.join(rgb_videos_path, category, video))
            
            # get all salmap images, move to salmap_videos/category/video
            tmp_salmap_path = os.path.join(salmap_videos_path, category, video)
            os.makedirs(tmp_salmap_path, exist_ok=True)
            # shutil.move(os.path.join(video_path, "salmap", "*"), os.path.join(salmap_videos_path, category, video))
            
            # get all fixation.mat, move to coordinate/category/video
            tmp_coordinate_path = os.path.join(coordinate_path, category, video)
            os.makedirs(tmp_coordinate_path, exist_ok=True)
            # shutil.move(os.path.join(video_path, "fixation", "maps", "*"), os.path.join(coordinate_path, category, video))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
            "--input", default="/home/elias/thesis/data/DADA-2000/DADA2000", help="")
    parser.add_argument(
        "--output", default="/home/elias/thesis/data/DADA-2000/DADA2000", help="")
    args = parser.parse_args()

    main(args.input, args.output)

