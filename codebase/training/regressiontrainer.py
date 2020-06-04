"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import sys

import torch
from tqdm import tqdm

from codebase.training.basetrainer import BaseTrainer


class RegressionTrainer(BaseTrainer):

    def __init__(self, cfg: dict):
        super(RegressionTrainer, self).__init__(cfg=cfg)

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.logger.train()

        # process bar handle
        pbar = tqdm(self.loader, file=sys.stdout)
        pbar.set_description(f'# Epoch {epoch}')

        # Iterate in batches over training set
        for data in pbar:

            # unwrapped data and send to device
            x_d, x_s, x_one_hot, q_stds, y = data
            x_d, x_s, q_stds = x_d.to(self.device), x_s.to(self.device), q_stds.to(self.device)
            x_one_hot, y = x_one_hot.to(self.device), y.to(self.device)

            # get predictions
            y_hat = self.model(x_d=x_d, x_s=x_s, x_one_hot=x_one_hot)[0]

            # calculate loss only over last_n predictions
            y_sub = y[:, -self.cfg["predict_last_n"]:, :]
            y_hat_sub = y_hat[:, -self.cfg["predict_last_n"]:, :]

            loss = self.loss_obj(y_hat_sub, y_sub, q_stds=q_stds)

            # delete old gradients
            self.optimizer.zero_grad()

            # get gradients
            loss.backward()

            if self.cfg.get("clip_gradient_norm", None) is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip_gradient_norm"])

            # update weights
            self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            self.logger.log_step(loss=loss.item())
