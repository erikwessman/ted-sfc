import os
import argparse
import scipy.io
import numpy as np


def main(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print("Path does not exist")
        exit(1)

    coordinates = {}

    for category in os.listdir(input_path):
        category_path = os.path.join(input_path, category)
        if not os.path.isdir(category_path):
            continue

        coordinates[category] = {}

        for video in os.listdir(category_path):
            video_path = os.path.join(category_path, video)
            if not os.path.isdir(video_path):
                continue

            maps_path = os.path.join(video_path, "fixation/maps")

            for file in os.listdir(maps_path):
                file_path = os.path.join(maps_path, file)
                if not os.path.isfile(file_path):
                    continue

                coordinates[category][video] = mat_to_coordinate(file_path)


def mat_to_coordinate(file_path):
    mat = scipy.io.loadmat(file_path)['I']
    non_zero_coords = np.argwhere(mat != 0)
    avg_x = np.mean(non_zero_coords[:, 0])
    avg_y = np.mean(non_zero_coords[:, 1])
    
    print(avg_x, avg_y)
    exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
            "--input", default="/home/elias/thesis/data/DADA-2000/DADA2000", help="")
    parser.add_argument(
        "--output", default="/home/elias/thesis/data/DADA-2000/DADA2000", help="")
    args = parser.parse_args()

    main(args.input, args.output)
