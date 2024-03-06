import os
from datetime import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import zCurve as z

OUTPUT_PATH = "./output/cut_in"


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


def plot_cells(df, output_path, display_plots):
    # Plot cut-in data.

    # We print the column headers to know what we have access to.
    cols = df.columns.values.tolist()

    # Next, we do subplots to plot the subcells.
    _, axs = plt.subplots(2, 3)
    axs[0, 0].plot(df["frame_id"], df["cell1"])
    axs[0, 0].set_title("cell1")
    axs[0, 1].plot(df["frame_id"], df["cell2"], "tab:orange")
    axs[0, 1].set_title("cell2")
    axs[0, 2].plot(df["frame_id"], df["cell3"], "tab:orange")
    axs[0, 2].set_title("cell3")
    axs[1, 0].plot(df["frame_id"], df["cell4"], "tab:green")
    axs[1, 0].set_title("cell4")
    axs[1, 1].plot(df["frame_id"], df["cell5"], "tab:red")
    axs[1, 1].set_title("cell5")
    axs[1, 2].plot(df["frame_id"], df["cell6"], "tab:red")
    axs[1, 2].set_title("cell6")

    for ax in axs.flat:
        ax.set(xlabel="frame_id", ylabel="delta angle")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(output_path, "cell_values.png"))
    if (display_plots):
        plt.show()


def plot_morton_codes(df, output_path, display_plots):
    data = df["morton"] / 1000000000000
    plt.figure()
    plt.xlabel("Morton")
    plt.ylabel("frequency")
    plt.ylim((0, 1))
    plt.eventplot(data, orientation="horizontal", colors="b", lineoffsets=0.5)

    plt.savefig(os.path.join(output_path, "morton_codes.png"))
    if (display_plots):
        plt.show()


def plot_red_strips_or_smth(df, output_path, display_plots):
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
    if (display_plots):
        plt.show()


def main(args):
    cell_values = pd.read_csv(args.csv_path, sep=";")

    output_path = os.path.join(
        OUTPUT_PATH,
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(output_path)

    cell_values, morton_codes = compute_morton_codes_for_cells(cell_values)

    plot_cells(cell_values, output_path, args.display_plots)
    plot_morton_codes(morton_codes, output_path, args.display_plots)
    plot_red_strips_or_smth(cell_values, output_path, args.display_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--csv-path", required=True, help="Path to the csv file with the cell values"
    )
    parser.add_argument("--display-plots", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(args)
