import os
import argparse
import pandas as pd
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
        morton_frame_pairs.append(
            {"frame_id": int(row["frame_id"]), "morton": morton_code}
        )

    return pd.DataFrame(morton_frame_pairs, columns=["frame_id", "morton"])


def main(data_path: str, display_plots: bool = False):
    for video_path, video_id, tqdm_obj in helper.traverse_videos(data_path):
        cell_value_path = os.path.join(video_path, "cell_values.csv")

        if not os.path.isfile(cell_value_path):
            tqdm_obj.write(f"Skipping {video_id}: Cell value CSV does not exist")
            continue

        cell_values = pd.read_csv(os.path.join(video_path, "cell_values.csv"), sep=";")
        morton_codes = compute_morton_codes_for_cells(cell_values)

        # The values get very big, reduce the size
        morton_codes["morton"] = morton_codes["morton"].div(10000000000)
        morton_codes.to_csv(f"{video_path}/morton_codes.csv", sep=";", index=False)

        plot_path = os.path.join(video_path, "plots")
        os.makedirs(plot_path, exist_ok=True)

        helper.create_and_save_CSP(morton_codes, plot_path, display_plots)
        helper.create_and_save_combined_plot_with_morton_codes(
            cell_values,
            morton_codes,
            plot_path,
            display_plots,
            "Cell values with Morton codes",
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path, args.display_plots)
