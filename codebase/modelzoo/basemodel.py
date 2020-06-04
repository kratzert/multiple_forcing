"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, cfg: dict):
        super(BaseModel, self).__init__()
        self.cfg = cfg

        self.output_size = len(cfg["target_variable"])
        self.head = cfg.get('head', 'regression')

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor,
                x_one_hot: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        raise NotImplementedError