"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDatasetH5(Dataset):

    def __init__(self, cfg: dict):
        self.h5_file = cfg["h5_file"]
        self.scaler_file = cfg["scaler_file"]
        self.cfg = cfg

        # get dictionary of feature scaler
        with self.scaler_file.open("rb") as fp:
            self.scaler = pickle.load(fp)

        # preload data if cached is true
        if self.cfg["cache_data"]:
            self.x_d, self.x_s, self.y, self.sample_2_basin, self.q_stds = self._preload_data()

        # read training basins from h5 file
        self.basins = self._get_basins()

        # if basin_id encoding, create random permuted id to int dictionary
        if cfg["use_basin_id_encoding"]:
            self.one_hot = torch.from_numpy(np.zeros(len(self.basins), dtype=np.float32))
            self.id_to_int = {b: i for i, b in enumerate(np.random.permutation(self.basins))}

            # dump id_to_int dictionary into run directory for validation
            file_path = self.cfg["train_dir"] / "id_to_int.p"
            if not file_path.is_file():
                with file_path.open("wb") as fp:
                    pickle.dump(self.id_to_int, fp)
            else:
                raise RuntimeError(f"File already exist at {file_path}")

        # has to be implemented in child class, returns None in case of no attributes
        self.df_attributes = self._load_attributes()

        # determine number of samples once
        if self.cfg["cache_data"]:
            self.num_samples = self.y.shape[0]
        else:
            with h5py.File(self.h5_file, 'r') as f:
                self.num_samples = f["target_data"].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.cfg["cache_data"]:
            x_d, x_s, y, basin, q_std = self._get_cached_sample(idx=idx)
        else:
            x_d, x_s, y, basin, q_std = self._get_h5_sample(idx=idx)

        if self.df_attributes is not None:
            # get attributes from data frame and create 2d array with copies
            x_attr = self.df_attributes.loc[self.df_attributes.index == basin].values.flatten()
            x_attr = torch.from_numpy(x_attr.astype(np.float32))
        else:
            x_attr = None

        if self.cfg["use_basin_id_encoding"]:
            x_one_hot = self.one_hot.zero_()
            x_one_hot[self.id_to_int[basin]] = 1
        else:
            x_one_hot = torch.empty(0)

        if (x_s is not None) and (x_attr is not None):
            x_s = torch.cat([x_attr, x_s], dim=-1)
        elif x_attr is not None:
            x_s = x_attr
        elif x_s is not None:
            pass
        else:
            x_s = torch.empty(0)

        return x_d, x_s, x_one_hot, q_std, y

    def _get_cached_sample(self, idx: int):
        x_d = self.x_d[idx]
        if self.x_s is not None:
            x_s = self.x_s[idx]
        else:
            x_s = None
        y = self.y[idx]
        basin = self.sample_2_basin[idx]
        q_std = self.q_stds[idx]

        return x_d, x_s, y, basin, q_std

    def _get_h5_sample(self, idx: int):
        with h5py.File(self.h5_file, 'r') as f:
            x_d = torch.from_numpy(f["dynamic_inputs"][idx].astype(np.float32))
            if "static_inputs" in f.keys():
                x_s = torch.from_numpy(f["static_inputs"][idx].astype(np.float32))
            else:
                x_s = None
            y = torch.from_numpy(f["target_data"][idx].astype(np.float32))
            basin = f["sample_2_basin"][idx]
            basin = basin.decode("ascii")
            q_std = torch.from_numpy(f["q_stds"][idx].astype(np.float32))

        return x_d, x_s, y, basin, q_std

    def _preload_data(self):
        with h5py.File(self.h5_file, 'r') as f:
            x_d = torch.from_numpy(f["dynamic_inputs"][:].astype(np.float32))
            y = torch.from_numpy(f["target_data"][:].astype(np.float32))
            str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            q_stds = torch.from_numpy(f["q_stds"][:].astype(np.float32))
            if "static_inputs" in f.keys():
                x_s = torch.from_numpy(f["static_inputs"][:].astype(np.float32))
            else:
                x_s = None
        return x_d, x_s, y, str_arr, q_stds

    def _get_basins(self):
        if self.cfg["cache_data"]:
            basins = list(set(self.sample_2_basin))
        else:
            with h5py.File(self.h5_file, 'r') as f:
                str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            basins = list(set(str_arr))
        return basins

    def _load_attributes(self):
        raise NotImplementedError
