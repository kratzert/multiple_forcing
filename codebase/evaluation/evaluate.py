"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from pathlib import PosixPath

from codebase.evaluation.tester import Tester


def start_evaluation(cfg: dict, run_dir: PosixPath, epoch: int):
    tester = Tester(cfg=cfg, run_dir=run_dir, mode="test")
    tester.evaluate(epoch=epoch, save_results=True, metrics=cfg.get('metrics', []))
