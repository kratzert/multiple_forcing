"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import torch

import codebase.training.loss as loss


def get_optimizer(model: torch.nn.Module, cfg: {}) -> torch.optim.Optimizer:
    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"][0])
    else:
        raise NotImplementedError(cfg["optimizer"])

    return optimizer


def get_loss_obj(cfg: dict) -> torch.nn.Module:
    if cfg["loss"] == "NSE":
        loss_obj = loss.MaskedNSELoss()
    elif cfg["loss"] == "MSE":
        loss_obj = loss.MaskedMSELoss()
    else:
        raise NotImplementedError(cfg["loss"])

    return loss_obj
