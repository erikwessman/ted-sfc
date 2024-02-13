import os
import argparse
import pandas as pd


def main(input_path: str, output_path: str):
    """
    Take the DADA-2000 dataset annotation spreadsheet and convert it to a
    format compatible with DRIVE, e.g. a text file with the format:
    1/001 1 0 100 50
    """
    if not os.path.isfile(input_path):
        print("Input file does not exist, exiting...")
        exit(1)

    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    df = trim_dataframe(df)

    export_dataframe(df, output_path)
    print("Done")


def trim_dataframe(df: pd.DataFrame):
    """Keep only the relevant columns and rename them"""
    return df[["type", "video", "whether an accident occurred (1/0)", "abnormal start frame", "abnormal end frame", "accident frame"]].rename(columns={
        "type": "type",
        "video": "id",
        "whether an accident occurred (1/0)": "accident",
        "abnormal start frame": "start",
        "abnormal end frame": "end",
        "accident frame": "toa",
    })


def export_dataframe(df: pd.DataFrame, output_path: str):
    with open(output_path, 'w') as file:
        for index, row in df.iterrows():
            padded_video_id = str(row['id']).zfill(3)
            formatted_line = f"{row['type']}/{padded_video_id} {row['accident']} {row['start']} {row['end']} {row['toa']}\n"
            file.write(formatted_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input", default="/home/erik/devel/ted-sfc/dada2k/data/DADA-annotation.xlsx", help="")
    parser.add_argument(
        "--output", default="/home/erik/devel/ted-sfc/dada2k/data/DADA-annotation.txt", help="")
    args = parser.parse_args()

    main(args.input, args.output)
