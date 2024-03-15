import os
from datetime import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import zCurve as z


def calculateMortonFrom6D_with_zCurve(a, b, c, d, e, f):
    # Cap floating point numbers to one decimal place
    a_int = int(round(a, 1) * 10)
    b_int = int(round(b, 1) * 10)
    c_int = int(round(c, 1) * 10)
    d_int = int(round(d, 1) * 10)
    e_int = int(round(e, 1) * 10)
    f_int = int(round(f, 1) * 10)
    value = z.interlace(a_int, b_int, c_int, d_int, e_int, f_int, dims=6)
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
        morton_code = calculateMortonFrom6D_with_zCurve(
            *(row[col] for col in cell_columns)
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


def main(args):
    output_path = args.output_path

    if not os.path.exists(output_path):
        raise ValueError(f"Output path {output_path} does not exist.")

    video_dirs = [
        name
        for name in os.listdir(output_path)
        if os.path.isdir(os.path.join(output_path, name))
    ]

    for video_id in video_dirs:
        print(f"Processing folder {video_id}")
        target_path = os.path.join(output_path, video_id)

        if os.path.isfile(os.path.join(target_path, "cell_values.csv")):
            cell_values = pd.read_csv(
                os.path.join(target_path, "cell_values.csv"), sep=";"
            )

            cell_values, morton_codes = compute_morton_codes_for_cells(cell_values)

            create_and_save_morton_codes(morton_codes, target_path, args.display_plots)
            create_and_save_red_stripes(cell_values, target_path, args.display_plots)
            print(f"Done. Results saved in {target_path}.")
        else:
            print(f"Skipped. File 'cell_values.csv' not found in {target_path}.")

        print("--------------------------")

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "output_path",
        help="Path to the directory containing the output for each video, including cell_values.csv.",
    )
    parser.add_argument("--display_plots", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(args)
