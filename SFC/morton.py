import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import zCurve as z

import helper


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the directory containing the output for each video, including cell_values.csv.",
    )
    parser.add_argument("--display_plots", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def calculateMortonFromList_with_zCurve(values):
    # Cap floating point numbers to one decimal place and convert to integers
    int_values = [int(round(value, 1) * 10) for value in values]
    value = z.interlace(*int_values, dims=len(int_values))
    return value


def calculateMortonFrom1D_with_zCurve(a):
    a_int = int(round(a, 1) * 10)
    return z.interlace(a_int)


def compute_morton_codes_for_cells(df):
    df = df.reset_index(drop=True)

    cell_columns = [col for col in df.columns if col.startswith("cell")]

    morton_frame_pairs = []
    for _, row in df.iterrows():
        # Generate a Morton code from the cell values in the row.
        morton_code = calculateMortonFromList_with_zCurve(
            (row[col] for col in cell_columns)
        )
        morton_frame_pairs.append({"morton": morton_code, "frame_id": row["frame_id"]})

    for col in cell_columns:
        df[f"{col}_morton"] = df[col].apply(calculateMortonFrom1D_with_zCurve)

    return df, pd.DataFrame(morton_frame_pairs, columns=["morton", "frame_id"])


def create_and_save_CSP(df, output_path, display_plots):
    plt.figure()
    plt.xlabel("Morton")
    plt.ylabel("frequency")
    plt.ylim((0, 1))
    plt.eventplot(df["morton"], orientation="horizontal", colors="b", lineoffsets=0.5)

    plt.savefig(os.path.join(output_path, "morton_codes.png"))

    if display_plots:
        plt.show()
    else:
        plt.close()


def create_and_save_CSP_with_dots(df, output_path, display_plots):
    fig, ax1 = plt.subplots()

    # Plot the blue lines
    ax1.set_xlabel("Morton")
    ax1.set_ylabel("Frequency")
    ax1.set_ylim((0, 1))
    ax1.eventplot([df["morton"].values], orientation="horizontal", colors="b", lineoffsets=0.5)

    # Plot one dot for each Morton code at each frame
    x = df["morton"]
    y = df["frame_id"]

    ax2 = ax1.twinx()
    ax2.scatter(x, y, s=5, color='black')
    ax2.set_ylabel('Frame Number')
    ax2.set_ylim([y.min(), y.max()])

    plt.savefig(os.path.join(output_path, "morton_codes_with_dots.png"))

    if display_plots:
        plt.show()
    else:
        plt.close()


def main(data_path, display_plots):
    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        cell_value_path = os.path.join(video_path, "cell_values.csv")

        if not os.path.isfile(cell_value_path):
            tqdm_obj.write(f"Skipping {video_id}: Cell value CSV does not exist")
            continue

        cell_values = pd.read_csv(os.path.join(video_path, "cell_values.csv"), sep=";")
        cell_values, morton_codes = compute_morton_codes_for_cells(cell_values)

        # The values get very big, reduce the size
        morton_codes["morton"] = morton_codes["morton"].div(10000000000)
        morton_codes.to_csv(f"{video_path}/morton_codes.csv", sep=";")

        create_and_save_CSP(morton_codes, video_path, display_plots)
        create_and_save_CSP_with_dots(morton_codes, video_path, display_plots)

    print("morton.py completed.")


if __name__ == "__main__":
    args = parse_arguments()

    main(args.data_path, args.display_plots)
