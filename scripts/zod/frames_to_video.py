import os
import subprocess
import argparse
import random


FPS = 10
SCALE = "1280:720"


def process_sequences(sequences_path, output_path, mode, max_videos):
    os.makedirs(output_path, exist_ok=True)

    sequences_dir = os.path.join(sequences_path, "sequences")
    sequence_names = os.listdir(sequences_dir)

    if mode == "random":
        random.shuffle(sequence_names)

    processed_videos = 0
    for sequence_name in sequence_names:
        if processed_videos >= max_videos:
            break

        sequence_dir = os.path.join(sequences_dir, sequence_name)
        camera_front_blur_dir = os.path.join(sequence_dir, "camera_front_blur")

        if os.path.isdir(camera_front_blur_dir):
            sequence_output_dir = os.path.join(output_path, sequence_name)
            os.makedirs(sequence_output_dir, exist_ok=True)

            ffmpeg_command = [
                "ffmpeg",
                "-framerate",
                str(FPS),
                "-pattern_type",
                "glob",
                "-i",
                os.path.join(camera_front_blur_dir, "*.jpg"),
                "-vf",
                f"scale={SCALE}",
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "veryfast",
                os.path.join(sequence_output_dir, "output.avi"),
            ]

            subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Processed sequence {sequence_name}")
            processed_videos += 1
        else:
            print(f"Skipped {sequence_dir}, camera_front_blur directory not found.")


def main():
    parser = argparse.ArgumentParser(
        description="Process sequences of images into videos."
    )
    parser.add_argument(
        "sequences_path",
        type=str,
        help="Path to the top-level folder containing the 'sequences' subfolder.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output path to store the processed videos",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "random"],
        default="sequential",
        help="Order in which to process the sequences. Default is 'sequential'.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=0,
        help="Maximum number of videos to process. Default is 0 (process all).",
    )

    args = parser.parse_args()

    # If max_videos is not specified or specified as 0, process all videos
    max_videos = args.max_videos if args.max_videos > 0 else float("inf")

    process_sequences(args.sequences_path, args.output_path, args.mode, max_videos)


if __name__ == "__main__":
    main()
