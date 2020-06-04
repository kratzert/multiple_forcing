"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str)
parser.add_argument('--gpu_ids', type=int, nargs='+', required=True)
parser.add_argument('--runs_per_gpu', type=int, required=True)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--run_dir', type=str)

args = parser.parse_args()

if args.mode not in ["train", "evaluate"]:
    raise ValueError("--mode must be either 'train' or 'evaluate")

if (args.mode == 'train') and (args.config_dir is None):
    raise ValueError("In train mode you have to specify --config_dir")
elif (args.mode == 'evaluate') and (args.run_dir is None):
    raise ValueError("In train mode you have to specify --run_dir")

available_nodes = args.gpu_ids
n_parallel_runs = len(available_nodes) * args.runs_per_gpu
gpu_counter = np.zeros((len(available_nodes)), dtype=np.int)

if args.mode == "train":
    config_dir = Path(args.config_dir)
    processes = list(config_dir.glob('**/*.yml'))
else:
    run_dir = Path(args.run_dir)
    processes = list(run_dir.glob('*'))

# for approximately equal memory usage during hyperparam tuning, randomly shuffle list of processes
random.shuffle(processes)

running_processes = {}

counter = 0

while True:

    # start new runs
    for _ in range(n_parallel_runs - len(running_processes)):

        if counter >= len(processes):
            break

        node_id = np.argmin(gpu_counter)
        gpu_counter[node_id] += 1
        gpu_id = available_nodes[node_id]
        process = processes[counter]

        environment = [f"export CUDA_VISIBLE_DEVICES={gpu_id};"]
        if args.mode == "train":
            command = [f"python main.py train --config_file {process}"]
        else:
            command = [f"python main.py evaluate --run_dir {process}"]

        run_cmd = ' '.join(environment + command)

        print(f"Starting run {counter+1}/{len(processes)}: {run_cmd}")

        running_processes[(run_cmd, node_id)] = subprocess.Popen(run_cmd, stdout=subprocess.DEVNULL, shell=True)
        counter += 1
        time.sleep(2)

    # check for completed runs
    for key, process in running_processes.items():
        if process.poll() is not None:
            print(f"Finished run {key[0]}")
            gpu_counter[key[1]] -= 1
            print("Cleaning up...\n\n")
            try:
                _ = process.communicate(timeout=5)
            except TimeoutError:
                print('')
                print("WARNING: PROCESS {} COULD NOT BE REAPED!".format(key))
                print('')
            running_processes[key] = None

    # delete possibly finished runs
    running_processes = {key: val for key, val in running_processes.items() if val is not None}
    time.sleep(2)

    if (len(running_processes) == 0) and (counter >= len(processes)):
        break

print("Done")
sys.stdout.flush()
