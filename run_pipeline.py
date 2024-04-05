import os
import sys
import subprocess
import argparse
import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to run data processing pipeline.")

    parser.add_argument("data_path", help="Path to the data directory.")
    parser.add_argument("output_path", help="Path where output will be saved.")
    parser.add_argument("dataset_config_path", help="Path to the dataset configuration file.")
    parser.add_argument("event_config_path", help="Path to the event configuration file.")
    parser.add_argument("--heatmap", help="Skip heatmap generation.", action=argparse.BooleanOptionalAction)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--attention", help="Use attention mechanism.", action=argparse.BooleanOptionalAction)
    group.add_argument("--optical-flow", help="Use optical flow.", action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def find_env_path(env_name):
    """
    Finds the path to the specified Conda environment.
    """
    try:
        envs = subprocess.check_output(["conda", "env", "list"], universal_newlines=True)
        for line in envs.splitlines():
            line_parts = line.split()
            if env_name == line_parts[0]:
                return line_parts[-1]
    except subprocess.CalledProcessError as e:
        print(f"Failed to list Conda environments. Error: {e}")
        sys.exit(1)
    return None


def run_script_in_conda_env(script_path, args, env_name):
    """
    Runs a Python script in the specified Conda environment using its path.
    """
    env_path = find_env_path(env_name)
    if not env_path:
        print(f"Conda environment '{env_name}' not found.")
        sys.exit(1)

    command = ['conda', 'run', '-p', env_path, 'python', script_path] + args
    try:
        # TODO print the output in real time
        subprocess.run(command, check=True, stdout=None, stderr=None)
    except subprocess.CalledProcessError:
        print(f"Failed to run {script_path} in the {env_name} environment.")
        sys.exit(1)


def check_path_exists(path, path_type):
    """
    Checks if the given path exists. Exits the program if it does not.
    """
    if not os.path.exists(path):
        print(f"{path_type} path '{path}' does not exist.")
        sys.exit(1)


def log_runtime(output_path, start_time, end_time, duration):
    """
    Logs the pipeline's runtime information to a text file.
    """
    log_file_path = os.path.join(output_path, "pipeline_runtime_log.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Pipeline Run: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Start Time: {start_time}\n")
        log_file.write(f"End Time: {end_time}\n")
        log_file.write(f"Total Runtime: {duration}\n")
        log_file.write("--------------------------------------------------\n")


def main(data_path, output_path, dataset_config_path, event_config_path, heatmap, attention, optical_flow):
    check_path_exists(data_path, "Data")
    check_path_exists(dataset_config_path, "Dataset config")
    check_path_exists(event_config_path, "Event config")

    if os.path.isdir(output_path):
        response = input("Output path already exists. Do you want to overwrite it? (y/[n]): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            sys.exit(1)
    else:
        print("Output path does not exist. Creating it.")
        os.makedirs(output_path, exist_ok=True)

    start_time = datetime.datetime.now()
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if heatmap and attention:
        print("----------------------------------------")
        print("Starting main_saliency.py...")
        print("----------------------------------------")
        run_script_in_conda_env(
            script_path="DRIVE/main_saliency.py",
            args=[data_path, output_path, dataset_config_path],
            env_name="pyRL"
        )

    if attention:
        print("----------------------------------------")
        print("Starting grid_attention.py...")
        print("----------------------------------------")
        run_script_in_conda_env(
            script_path="SFC/grid_attention.py",
            args=[data_path, output_path, dataset_config_path, event_config_path],
            env_name="TED-SFC"
        )

    if optical_flow:
        print("----------------------------------------")
        print("Starting grid_optical_flow.py...")
        print("----------------------------------------")
        run_script_in_conda_env(
            script_path="SFC/grid_optical_flow.py",
            args=[data_path, output_path, dataset_config_path, event_config_path],
            env_name="TED-SFC"
        )

    print("----------------------------------------")
    print("Starting morton.py...")
    print("----------------------------------------")
    run_script_in_conda_env(
        script_path="SFC/morton.py",
        args=[output_path],
        env_name="TED-SFC"
    )

    print("========================================")
    print(f"Pipeline completed. Output saved to '{output_path}'.")
    print("========================================")

    end_time = datetime.datetime.now()
    print(f"Pipeline ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total runtime: {hours}h {minutes}m {seconds}s")

    # Log the runtime information
    log_runtime(args.output_path, start_time, end_time, f"{hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    args = parse_arguments()

    if args.optical_flow:
        args.heatmap = False

    main(args.data_path, args.output_path, args.dataset_config_path, args.event_config_path, args.heatmap, args.attention, args.optical_flow)
