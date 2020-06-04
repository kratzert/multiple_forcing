"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from codebase.errors import NoTrainDataError
from codebase.data.utils import reshape_data


class BaseDatasetBasin(Dataset):

    def __init__(self,
                 basin: str,
                 cfg: dict,
                 mode: str,
                 additional_features: List[pd.DataFrame] = [],
                 id_to_int: dict = {},
                 scaler: dict = {}):
        super(BaseDatasetBasin, self).__init__()
        self.basin = basin
        self.cfg = cfg

        if mode not in ['train', 'validation', 'test']:
            raise ValueError(f"'mode' has to be one of ['train', 'validation', 'test']")

        self.mode = mode
        self.is_train = True if mode == "train" else False
        self.additional_features = additional_features
        self.id_to_int = id_to_int
        self.scaler = scaler

        self.data_dir = cfg["data_dir"]
        self.seq_length = cfg["seq_length"]

        # Make sure all dates (even a single date) is a list so we can iterate over it when slicing the data frame
        if isinstance(cfg[f'{mode}_start_date'], list):
            if not self.is_train:
                raise ValueError(
                    "Only one continuous period for evaluation is currently supported, i.e. use a single start date")
            self.start_dates = cfg[f'{mode}_start_date']
        else:
            self.start_dates = [cfg[f'{mode}_start_date']]
        if isinstance(cfg[f'{mode}_end_date'], list):
            if not self.is_train:
                raise ValueError(
                    "Only one continuous period for evaluation is currently supported, i.e. use a single start date")
            self.end_dates = cfg[f'{mode}_end_date']
        else:
            self.end_dates = [cfg[f'{mode}_end_date']]

        if (cfg["dynamic_inputs"] is None) or (not cfg["dynamic_inputs"]):
            raise ValueError("At least one feature has to be specified as dynamic input")
        self.dynamic_inputs = cfg["dynamic_inputs"]

        if (cfg["target_variable"] is None) or (not cfg["target_variable"]):
            raise ValueError("At least one target feature has to be specified")
        self.target_variable = cfg["target_variable"]

        if (cfg["static_inputs"] is None) or (not cfg["static_inputs"]):
            self.static_inputs = []
        else:
            self.static_inputs = cfg["static_inputs"]

        #place holder for data arrays
        self.attributes = None
        self.x_d = None
        self.x_s = None
        if (not self.is_train) and (self.id_to_int):
            self.one_hot = torch.FloatTensor(len(self.id_to_int))
            self.one_hot[id_to_int[basin]] = 1.0
        else:
            self.one_hot = torch.empty(0)
        self.y = None
        self.num_samples = None

        # placeholder to store std of discharge, used for rescaling losses during training
        self.q_std = None

        # placeholder to store start and end date of entire period (incl warmup)
        self.period_start = None
        self.period_end = None

    def _load_data(self):
        raise NotImplementedError

    def _preprocess_data(self):

        df = self._load_data()

        # Merge CAMELS data with additionally passed features in list of dataframes
        if self.additional_features:
            df = pd.concat([df, *self.additional_features], axis=1)

        x_d_list, x_s_list, y_list = [], [], []

        for start_date, end_date in zip(self.start_dates, self.end_dates):
            # we use (seq_len) time steps before start for warmup
            warmup_start_date = start_date - pd.DateOffset(days=self.seq_length - 1)
            df_sub = df[warmup_start_date:end_date]

            # store first and last date of the selected period (including warm_start), needed for validation/testing
            self.period_start = df_sub.index[0]
            self.period_end = df_sub.index[-1]

            x_d = self._get_feature_array(df=df_sub, features=self.dynamic_inputs)
            y = self._get_feature_array(df=df_sub, features=self.target_variable)

            if np.isnan(y).sum() == y.size:
                raise NoTrainDataError("Basin contains no valid discharge observations in selected period.")

            if self.static_inputs:
                x_s = self._get_feature_array(df=df_sub, features=self.static_inputs)
            else:
                x_s = None

            # normalize data, reshape for LSTM training and remove invalid samples
            if not self.is_train:
                x_d = (x_d - self.scaler["dyn_mean"]) / self.scaler["dyn_std"]
                if x_s is not None:
                    x_s = (x_s - self.scaler["stat_mean"]) / self.scaler["stat_std"]

            x_d, x_s, y = reshape_data(x_d=x_d, x_s=x_s, y=y, seq_length=self.seq_length)
            x_d_list.append(x_d)
            y_list.append(y)
            if x_s is not None:
                x_s_list.append(x_s)

        x_d = np.concatenate(x_d_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        if x_s_list:
            x_s = np.concatenate(x_s_list, axis=0)

        if self.is_train:

            if np.sum(np.isnan(x_d)) > 0:
                # get set of unique samples, where any input contains a NaN
                idx = list(set(np.argwhere(np.isnan(x_d))[:, 0]))

                # delete those samples
                y = np.delete(y, idx, axis=0)
                x_d = np.delete(x_d, idx, axis=0)
                if x_s is not None:
                    x_s = np.delete(x_s, idx, axis=0)

            # get statistics
            # We don't calculate the statistics of the x_s here, since some of these could be
            # really static over time, so the std would be 0, which is problematic when combining
            # basins.
            x_cat = np.concatenate([x_d[0, :, :], x_d[1:, -1, :]], axis=0)
            self.scaler["dyn_mean"] = np.nanmean(x_cat, axis=0)
            self.scaler["dyn_std"] = np.nanstd(x_cat, axis=0)
            y_cat = np.concatenate([y[0, :, :], y[1:, -1, :]], axis=0)
            self.scaler["target_mean"] = np.nanmean(y_cat, axis=0)
            self.scaler["target_std"] = np.nanstd(y_cat, axis=0)

            # check if qobs in target variable, if yes store std for NSELoss separately
            qobs_var = [v for v in self.target_variable if "qobs" in v.lower()]
            if qobs_var:
                qobs_pos = self.target_variable.index(qobs_var[0])
                self.q_std = self.scaler["target_std"][qobs_pos]

        # convert arrays to torch tensors
        x_d = torch.from_numpy(x_d.astype(np.float32))
        if x_s is not None:
            x_s = torch.from_numpy(x_s.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x_d, x_s, y

    def get_scaler(self):
        return self.scaler

    @staticmethod
    def _get_feature_array(df, features):
        if any([feat not in df.columns for feat in features]):
            raise RuntimeError("Not all features specified match the available dataframe columns")
        else:
            x = np.array([df[feature].values for feature in features]).T
        return x

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if (self.attributes is not None) and (self.x_s is not None):
            x_s = torch.cat([self.attributes, self.x_s[idx]], dim=-1)
        elif self.attributes is not None:
            x_s = self.attributes
        elif self.x_s is not None:
            x_s = self.x_s[idx].unsqueeze(0)
        else:
            x_s = torch.empty(0)

        return self.x_d[idx], x_s, self.one_hot, self.y[idx]
