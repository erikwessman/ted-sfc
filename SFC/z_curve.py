import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import zCurve as z
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "output_path",
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


def create_and_save_morton_codes(df, output_path, display_plots):
    data = df["morton"] / 1000000000000
    plt.figure()
    plt.xlabel("Morton")
    plt.ylabel("frequency")
    plt.ylim((0, 1))
    plt.eventplot(data, orientation="horizontal", colors="b", lineoffsets=0.5)

    plt.savefig(os.path.join(output_path, "morton_codes.png"))
    if display_plots:
        plt.show()


def create_and_save_red_stripes(df, output_path, display_plots):
    colors = ["b", "orange", "orange", "g", "r", "r"]

    plt.figure()
    plt.xlabel("Morton")
    plt.ylabel("Frequency")
    plt.ylim((0, 1))

    # Dynamically generate event plots for each cell's Morton code.
    for i, col in enumerate(df.columns[df.columns.str.contains("_morton")]):
        data = df[col] / 1000000000000
        plt.eventplot(
            data,
            orientation="horizontal",
            colors=colors[i % len(colors)],
            lineoffsets=0.5,
        )

    plt.savefig(os.path.join(output_path, "morton_codes_event_plot.png"))
    if display_plots:
        plt.show()


def main(output_path, display_plots):
    assert os.path.exists(output_path), f"Output path {output_path} does not exist."

    video_dirs = [
        name
        for name in os.listdir(output_path)
        if os.path.isdir(os.path.join(output_path, name))
    ]

    pbar = tqdm(video_dirs, desc="Processing folders")
    for video_id in pbar:
        pbar.set_description(f"Processing folder {video_id}")
        target_path = os.path.join(output_path, video_id)

        if os.path.isfile(os.path.join(target_path, "cell_values.csv")):
            cell_values = pd.read_csv(
                os.path.join(target_path, "cell_values.csv"), sep=";"
            )

            cell_values, morton_codes = compute_morton_codes_for_cells(cell_values)

            create_and_save_morton_codes(morton_codes, target_path, display_plots)
            create_and_save_red_stripes(cell_values, target_path, display_plots)
        else:
            tqdm.write(f"Skipped. File 'cell_values.csv' not found in {target_path}.")

        pbar.set_description("Processing folders")


if __name__ == "__main__":
    args = parse_arguments()

    main(args.output_path, args.display_plots)
