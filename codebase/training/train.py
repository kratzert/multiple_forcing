"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from codebase.training.regressiontrainer import RegressionTrainer


def start_training(cfg: dict):

    if cfg["head"] == 'regression':
        trainer = RegressionTrainer(cfg=cfg)
    else:
        raise ValueError(f"Unknown head {cfg['head']} or Trainer not configured")

    trainer.initialize_training()
    trainer.train_and_validate()
