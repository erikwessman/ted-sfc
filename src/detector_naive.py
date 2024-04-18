"""
This script attempts to find the event window in a video from a set of morton codes
"""

import os
import argparse
import pandas as pd
from typing import Union, Tuple

import helper


MAX_GAP_SEC = 2  # The max allowed gap between any two cells
MAX_LENGTH_SEC = 10  # The max allowed total event time
MIN_LENGTH_SEC = 1  # The min allowed total event time


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the directory containing the output for each video",
    )
    parser.add_argument(
        "config_path",
        help="Path to the config yml file",
    )

    return parser.parse_args()


def frames_to_seconds(nr_frames: int, fps: int):
    return nr_frames / fps


def is_valid_sequence(
    sequence, required_cell_subsets, cell_gap_time_limits, event_length_limit, fps
) -> bool:
    """
    Checks if a sequence of tuples, e.g. [(1, 10), (2, 20)] meets the
    requirements to be an event.
    The first element of the tuples represents the cell ID, and the second
    is the frame ID.
    """
    if len(sequence) < 2:
        return False

    total_nr_frames = abs(sequence[-1][1] - sequence[0][1])
    total_nr_seconds = frames_to_seconds(total_nr_frames, fps)

    # Sequence must be within the event length limit
    if not event_length_limit[0] <= total_nr_seconds <= event_length_limit[1]:
        return False

    # Sequence must contain at least one match from each required cell subset
    unique_cells = {s[0] for s in sequence}
    total_matches = 0
    for required_cell_subset in required_cell_subsets:
        count = len(required_cell_subset & unique_cells)
        total_matches += count
        if count < 1:
            return False

    # Must match with at least 3 unique cells
    if total_matches < 3:
        return False

    nr_frames_standing_still = 0

    for i in range(1, len(sequence)):
        prev_cell_key = sequence[i - 1][0] - 1
        curr_cell_key = sequence[i][0] - 1

        prev_frame = sequence[i - 1][1]
        curr_frame = sequence[i][1]

        cell_key_diff = curr_cell_key - prev_cell_key
        frame_diff = curr_frame - prev_frame

        # Don't allow sequence to make sudden jumps by more than 2 cells
        if abs(cell_key_diff) > 2:
            return False

        # Don't allow sequence to stand still for too long
        if nr_frames_standing_still > 20:
            pass

        if cell_key_diff > 0:
            nr_frames_standing_still = 0

            min_gap, max_gap = 0, 0
            for i in range(prev_cell_key, curr_cell_key):
                min_gap += cell_gap_time_limits[i][0]
                max_gap += cell_gap_time_limits[i][1]
        elif cell_key_diff < 0:
            nr_frames_standing_still = 0

            min_gap, max_gap = 0, 0
            for i in range(curr_cell_key, prev_cell_key - 1, -1):
                min_gap += cell_gap_time_limits[i][0]
                max_gap += cell_gap_time_limits[i][1]
        else:
            nr_frames_standing_still += abs(frame_diff)
            continue

        time_gap = frames_to_seconds(abs(curr_frame - prev_frame), fps)

        # The total time it took to go from prev_cell to curr_cell should not exceed specified limits
        if not min_gap <= time_gap <= max_gap:
            return False

    return True


def get_sequence_with_most_cells(sequences):
    max_unique_count = -1
    list_with_max_unique = []

    for sequence in sequences:
        unique_first_elements = len(set(x[0] for x in sequence))

        if unique_first_elements > max_unique_count:
            max_unique_count = unique_first_elements
            list_with_max_unique = sequence

    return list_with_max_unique


def get_ordered_sequences(data, tolerance=1):
    active_sequences = []
    final_sequences = []

    for current_match in data:
        new_active_sequences = []
        fit_any_sequence = False

        # Check against all active sequences
        for seq_info in active_sequences:
            sequence, interruptions = seq_info

            # Check if current tuple can extend the sequence
            if current_match[0] >= sequence[-1][0]:
                sequence.append(current_match)
                new_active_sequences.append((sequence, 0))  # reset non-increasing count
                fit_any_sequence = True
            else:
                # Increment non-increasing count if within tolerance
                if interruptions < tolerance:
                    new_active_sequences.append((sequence, interruptions + 1))
                    new_active_sequences.append(([current_match], 0))
                else:
                    # Finalize and remove sequence if tolerance exceeded
                    if len(sequence) > 1:
                        final_sequences.append(sequence)

        # If current doesn't fit any sequence and isn't a continuation, start a new one
        if not fit_any_sequence:
            new_active_sequences.append(([current_match], 0))

        # Update active sequences
        active_sequences = new_active_sequences

    # Finalize all remaining active sequences
    for sequence, _ in active_sequences:
        if len(sequence) > 1:
            final_sequences.append(sequence)

    return final_sequences


def get_sequence(cell_values, required_cell_subsets, cell_gap_time_limits, event_length_limit, fps):
    """ """
    path = []

    for _, row in cell_values.iterrows():
        curr_frame_id = row["frame_id"]
        non_zero_cell_value_indices = [
            int(cell.split("cell")[1]) for cell in row.index[1:] if row[cell] > 0
        ]

        if not non_zero_cell_value_indices:
            continue

        for cell_index in non_zero_cell_value_indices:
            path.append((cell_index, curr_frame_id))

    if not path:
        return None

    valid_sequences = []

    ordered_sequences = get_ordered_sequences(path)
    for sequence in ordered_sequences:
        if is_valid_sequence(
            sequence,
            required_cell_subsets,
            cell_gap_time_limits,
            event_length_limit,
            fps,
        ):
            valid_sequences.append(sequence)

    if valid_sequences:
        return get_sequence_with_most_cells(valid_sequences)
    else:
        return None


def detect_event(
    cell_values,
    required_cell_subsets,
    cell_gap_time_limits,
    event_length_limit,
    fps,
) -> Union[Tuple[int, int, str], None]:
    """
    Returns the event window frame interval and the type, e.g. (30, 50, "left")
    In case there is no event, returns: None
    """
    sequence_left = get_sequence(
        cell_values,
        required_cell_subsets,
        cell_gap_time_limits,
        event_length_limit,
        fps,
    )
    sequence_right = get_sequence(
        cell_values,
        required_cell_subsets,
        cell_gap_time_limits,
        event_length_limit,
        fps,
    )

    if sequence_left:
        return sequence_left[0][1], sequence_left[-1][1], "left"
    elif sequence_right:
        return sequence_right[-1][1], sequence_right[0][1], "right"
    else:
        return None


def main(data_path: str, config_path: str):
    # Load config
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]
    detector_config = config["detector_config"]

    # Grid variables
    fps = grid_config["fps"]

    # Detector variables
    required_cell_subsets = detector_config["required_cell_subsets"]
    cell_gap_time_limits = detector_config["cell_gap_time_limits"]
    event_length_limit = detector_config["event_length_limit_seconds"]

    df_event_window = pd.DataFrame(
        columns=["video_id", "event_detected", "start_frame", "end_frame"]
    )

    for _, video_id, tqdm_obj in helper.traverse_videos(data_path):
        target_path = os.path.join(data_path, video_id)

        if not os.path.isfile(os.path.join(target_path, "cell_values.csv")):
            tqdm_obj.write(
                f"Skipping {video_id}: cell_values.csv does not exist at {target_path}"
            )
            continue

        cell_values = pd.read_csv(os.path.join(target_path, "cell_values.csv"), sep=";")

        event = detect_event(
            cell_values,
            required_cell_subsets,
            cell_gap_time_limits,
            event_length_limit,
            fps,
        )

        if event:
            event_detected = True
            start_frame, end_frame, scenario_type = event
        else:
            event_detected = False
            start_frame, end_frame, scenario_type = -1, -1, ""

        df_event_window = df_event_window._append(
            {
                "video_id": video_id,
                "event_detected": event_detected,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "scenario_type": scenario_type,
            },
            ignore_index=True,
        )

    df_event_window.to_csv(
        os.path.join(data_path, "naive_event_window.csv"), sep=";", index=False
    )

    helper.save_config(detector_config, data_path, "detector_config.yml")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path, args.config_path)
