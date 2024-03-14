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
    # Make sure indexes pair with number of rows.
    df = df.reset_index(drop=True)

    lst = []
    for _, row in df.iterrows():
        morton_code = calculateMortonFrom6D_with_zCurve(
            row["cell1"],
            row["cell2"],
            row["cell3"],
            row["cell4"],
            row["cell5"],
            row["cell6"],
        )
        lst.append({"morton": morton_code, "frame_id": row["frame_id"]})

    df["cell1_morton"] = df["cell1"].apply(calculateMortonFrom1D_with_zCurve)
    df["cell2_morton"] = df["cell2"].apply(calculateMortonFrom1D_with_zCurve)
    df["cell3_morton"] = df["cell3"].apply(calculateMortonFrom1D_with_zCurve)
    df["cell4_morton"] = df["cell4"].apply(calculateMortonFrom1D_with_zCurve)
    df["cell5_morton"] = df["cell5"].apply(calculateMortonFrom1D_with_zCurve)
    df["cell6_morton"] = df["cell6"].apply(calculateMortonFrom1D_with_zCurve)
    return (df, pd.DataFrame(lst, columns=["morton", "frame_id"]))

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
    data_cell1 = df["cell1_morton"] / 1000000000000
    data_cell2 = df["cell2_morton"] / 1000000000000
    data_cell3 = df["cell3_morton"] / 1000000000000
    data_cell4 = df["cell4_morton"] / 1000000000000
    data_cell5 = df["cell5_morton"] / 1000000000000
    data_cell6 = df["cell6_morton"] / 1000000000000

    plt.figure()
    plt.xlabel("Morton")
    plt.ylabel("frequency")
    plt.ylim((0, 1))

    plt.eventplot(data_cell1, orientation="horizontal", colors="b", lineoffsets=0.5)
    plt.eventplot(
        data_cell2, orientation="horizontal", colors="orange", lineoffsets=0.5
    )
    plt.eventplot(
        data_cell3, orientation="horizontal", colors="orange", lineoffsets=0.5
    )
    plt.eventplot(data_cell4, orientation="horizontal", colors="g", lineoffsets=0.5)
    plt.eventplot(data_cell5, orientation="horizontal", colors="r", lineoffsets=0.5)
    plt.eventplot(data_cell6, orientation="horizontal", colors="r", lineoffsets=0.5)

    plt.savefig(os.path.join(output_path, "red_morton_codes.png"))
    if display_plots:
        plt.show()


def main(args):
    output_path = args.output_path

    if not os.path.exists(output_path):
        raise ValueError(f"Output path {output_path} does not exist.")

    video_dirs = [name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))]

    for video_id in video_dirs:
        print(f"Processing folder {video_id}")
        target_path = os.path.join(output_path, video_id)

        if os.path.isfile(os.path.join(target_path, "cell_values.csv")):
            cell_values = pd.read_csv(os.path.join(target_path, "cell_values.csv"), sep=";")

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
    parser.add_argument("--display-plots", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(args)
