"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from typing import Dict

import torch
import torch.nn as nn

from codebase.modelzoo.head import get_head
from codebase.modelzoo.basemodel import BaseModel


class CudaLSTM(BaseModel):

    def __init__(self, cfg: Dict):
        super(CudaLSTM, self).__init__(cfg=cfg)

        n_attributes = 0
        if ("camels_attributes" in cfg.keys()) and cfg["camels_attributes"]:
            n_attributes += len(cfg["camels_attributes"])

        self.input_size = len(cfg["dynamic_inputs"] + cfg.get("static_inputs", [])) + n_attributes
        if cfg["use_basin_id_encoding"]:
            self.input_size += cfg["number_of_basins"]

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=cfg["hidden_size"])

        self.dropout = nn.Dropout(p=cfg["output_dropout"])

        self.head = get_head(cfg=cfg, n_in=cfg["hidden_size"], n_out=self.output_size)

        self.reset_parameters()

    def reset_parameters(self):
        if self.cfg["initial_forget_bias"] is not None:
            hidden_size = self.cfg["hidden_size"]
            self.lstm.bias_hh_l0.data[hidden_size:2 * hidden_size] = self.cfg["initial_forget_bias"]

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor, x_one_hot: torch.Tensor):
        # transpose to [seq_length, batch_size, n_features]
        x_d = x_d.transpose(0, 1)

        # concat all inputs
        if (x_s.nelement() > 0) and (x_one_hot.nelement() > 0):
            x_s = x_s.unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_one_hot = x_one_hot.unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
        elif x_s.nelement() > 0:
            x_s = x_s.unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s], dim=-1)
        elif x_one_hot.nelement() > 0:
            x_one_hot = x_one_hot.unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_one_hot], dim=-1)
        else:
            pass

        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [batch_size, seq_length, n_hiddens]
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        y_hat = self.head(self.dropout(lstm_output.transpose(0, 1)))

        return y_hat, h_n, c_n
