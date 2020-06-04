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


def get_head(cfg: dict, n_in: int, n_out: int) -> nn.Module:
    if cfg["head"] == "regression":
        head = Regression(n_in=n_in, n_out=n_out, activation=cfg.get("output_activation", None))
    else:
        raise NotImplementedError(cfg["head"])

    return head


class Regression(nn.Module):
    """
    Regression head with different output activations.
    """

    def __init__(self, n_in: int, n_out: int, activation: str):
        super(Regression, self).__init__()

        layers = [nn.Linear(n_in, n_out)]
        if activation is not None:
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            elif activation.lower() == "linear":
                pass
            else:
                print(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)