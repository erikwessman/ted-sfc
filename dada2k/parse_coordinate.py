import os
import argparse


def main(input_path: str, output_path: str):

    # iterate over all categories

    # cd into each one

    # cd into fixation

    # cd into maps

    # iterate over .mat files

    # convert .mat to coordinates (x,y)

    # write to 00X_coordinate.txt

    if not os.path.exists(input_path):
        print("Path does not exist")
        exit(1)

    coordinates = {}

    for dir in os.listdir(input_path):
        dir_path = os.path.join(input_path, dir)
        if not os.path.isdir(dir_path):
            continue

        category_path = os.path.join(input_path, dir)

        maps_path = os.path.join(category_path, "fixation/maps")

        for file in os.listdir(maps_path):
            file_path = os.path.join(maps_path, file)
            if not os.path.isfile(file_path):
                continue

            coordinates[file] = mat_to_coordinate(file_path)

    pass


def mat_to_coordinate(file_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
            "--input", default="/mnt/sdb/DADA-2000/DADA2000", help="")
    parser.add_argument(
        "--output", default="/mnt/sdb/DADA-2000/DADA2000", help="")
    args = parser.parse_args()

    main(args.input, args.output)
