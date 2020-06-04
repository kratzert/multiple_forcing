"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import argparse
from pathlib import Path

from codebase.config import read_config
from codebase.evaluation.evaluate import start_evaluation
from codebase.training.train import start_training


def get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate"])
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--run_dir', type=str)
    parser.add_argument('--epoch', type=int, help="Epoch, of which the model should be evaluated")
    args = vars(parser.parse_args())

    if (args["mode"] == "train") and (args["config_file"] is None):
        raise ValueError("Missing path to config file")

    if (args["mode"] == "evaluate") and (args["run_dir"] is None):
        raise ValueError("Missing path to run directory")

    return args


def start_run(args: dict):

    config_file = Path(args["config_file"])
    config = read_config(config_file)

    print(f"### Run configurations for {config['experiment_name']}")
    for key, val in config.items():
        print(f"{key}: {val}")

    start_training(config)


def eval_run(args: dict):
    config_file = Path(args["run_dir"]) / "config.yml"
    config = read_config(config_file)

    start_evaluation(cfg=config, run_dir=Path(args["run_dir"]), epoch=args["epoch"])


if __name__ == "__main__":
    args = get_args()
    if args["mode"] == "train":
        start_run(args)
    elif args["mode"] == "evaluate":
        eval_run(args)
    else:
        raise RuntimeError(f"Unknown mode {args['mode']}")
