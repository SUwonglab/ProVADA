"""
run_multiple.py

Runs multiple provada runs on the same config file. Utilizes one GPU per run.
"""

import os
import argparse
import subprocess
import random
import time
import tempfile
from pathlib import Path


def get_parser():
    """
    Get the parser for the run_multiple.py script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--available_gpus", type=int, nargs="+", required=True)
    parser.add_argument("--reference_distribution_file", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sampler_type", type=str, required=False, default=None)
    return parser


def main():
    """
    Main function for the run_multiple.py script.
    """
    parser = get_parser()
    args = parser.parse_args()

    print(
        f"Launching {len(args.available_gpus)} runs on the following GPUs: {args.available_gpus}"
    )

    processes = []
    temp_output_files = []
    run_metadata = []

    # Create a logging_subdir name based on time
    logging_subdir = time.strftime("multiple_run_group_%Y-%m-%d_%H-%M-%S")

    # Set the random seed
    rng = random.Random(args.seed)

    # Launch a run for each visible device
    for job_idx, gpu in enumerate(args.available_gpus):
        # Set the visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Copy the env
        env = os.environ.copy()

        seed = rng.randint(0, 1000000)

        command = f"python run_provada.py --config {args.config} --device cuda:0 --seed {seed} --logging_subdir {logging_subdir}"

        if args.reference_distribution_file is not None:
            command += f" --reference_distribution_file {args.reference_distribution_file}"

        if args.sampler_type is not None:
            command += f" --sampler_type {args.sampler_type}"

        # Create temporary files to capture stdout/stderr
        temp_stdout = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.stdout')
        temp_stderr = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.stderr')
        temp_output_files.append((temp_stdout, temp_stderr))

        # Store metadata for this run
        config_name = Path(args.config).stem
        run_metadata.append({
            'gpu': gpu,
            'seed': seed,
            'config_name': config_name,
        })

        # Run the provada run
        # Always capture to temp files so we can check for early errors
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=temp_stdout,
            stderr=temp_stderr,
            env=env,
        )
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()

    print("All runs completed")

    # Check for failures and handle error logging
    for i, (process, (temp_stdout, temp_stderr), metadata) in enumerate(zip(processes, temp_output_files, run_metadata)):
        exit_code = process.returncode
        gpu = metadata['gpu']
        seed = metadata['seed']
        config_name = metadata['config_name']

        if exit_code != 0:
            # Check if a log file was created (indicates the run got far enough to create logs)
            log_dir = Path("logs") / logging_subdir
            log_files = list(log_dir.glob(f"*{seed}*.log")) if log_dir.exists() else []

            # If no log file exists, this was an early failure - create one
            if not log_files:
                # Create the log directory if it doesn't exist
                log_dir.mkdir(parents=True, exist_ok=True)

                # Create a log file name similar to what would have been created
                # Format: config_name_randomstring_seed.log
                log_filename = f"{config_name}_early_failure_{seed}.log"
                log_path = log_dir / log_filename

                # Write the captured output to the log file
                with open(log_path, 'w') as log_file:
                    log_file.write(f"=" * 80 + "\n")
                    log_file.write(f"EARLY FAILURE - Process exited with code {exit_code}\n")
                    log_file.write(f"GPU: {gpu}, Seed: {seed}\n")
                    log_file.write(f"Config: {args.config}\n")
                    log_file.write(f"Command: python run_provada.py --config {args.config} --device cuda:0 --seed {seed}\n")
                    log_file.write(f"=" * 80 + "\n\n")

                    # Read and write stderr (most likely to contain the error)
                    temp_stderr.seek(0)
                    stderr_content = temp_stderr.read()
                    if stderr_content.strip():
                        log_file.write("STDERR:\n")
                        log_file.write(stderr_content)
                        log_file.write("\n\n")

                    # Read and write stdout
                    temp_stdout.seek(0)
                    stdout_content = temp_stdout.read()
                    if stdout_content.strip():
                        log_file.write("STDOUT:\n")
                        log_file.write(stdout_content)
                        log_file.write("\n")

                print(f"Process {process.pid} (GPU {gpu}, seed {seed}) failed early (exit code {exit_code}) - error logged to: {log_path}")
            else:
                print(f"Process {process.pid} (GPU {gpu}, seed {seed}) exited with code {exit_code} - check log: {log_files[0]}")
        else:
            print(f"Process {process.pid} (GPU {gpu}, seed {seed}) completed successfully")

        # Clean up temp files
        temp_stdout.close()
        temp_stderr.close()
        os.unlink(temp_stdout.name)
        os.unlink(temp_stderr.name)


if __name__ == "__main__":
    main()
