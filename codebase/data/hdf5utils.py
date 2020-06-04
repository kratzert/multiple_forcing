"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import pickle
import sys
from pathlib import Path, PosixPath
from typing import Dict, List

import h5py
import numpy as np
from tqdm import tqdm

from codebase.errors import NoTrainDataError
from codebase.data import get_basin_dataset
from codebase.data.utils import get_camels_scaler, attributes_sanity_check

# used to perform in place normalization
CHUNK_SIZE = 5000


def create_h5_file(basins: List,
                   cfg: Dict,
                   h5_file: PosixPath,
                   scaler_file: PosixPath,
                   additional_features: List[Dict] = []):

    if h5_file.is_file():
        raise FileExistsError(f"File already exists at {h5_file}")

    if cfg.get("camels_attributes", []):
        attributes_sanity_check(data_dir=cfg["data_dir"],
                                dataset=cfg["dataset"],
                                basins=basins,
                                attribute_list=cfg.get("camels_attributes", []))

    n_dyn_inputs = len(cfg["dynamic_inputs"])
    n_targets = len(cfg["target_variable"])
    # we only store user-defined additional static features provided in the additional_features table
    n_stat = len(cfg["static_inputs"])

    with h5py.File(h5_file, 'w') as out_f:
        dyn_input_data = out_f.create_dataset('dynamic_inputs',
                                              shape=(0, cfg["seq_length"], n_dyn_inputs),
                                              maxshape=(None, cfg["seq_length"], n_dyn_inputs),
                                              chunks=True,
                                              dtype=np.float32,
                                              compression='gzip')
        if n_stat > 0:
            stat_input_data = out_f.create_dataset('static_inputs',
                                                   shape=(0, n_stat),
                                                   maxshape=(None, n_stat),
                                                   chunks=True,
                                                   dtype=np.float32,
                                                   compression='gzip')
        target_data = out_f.create_dataset('target_data',
                                           shape=(0, cfg["seq_length"], n_targets),
                                           maxshape=(None, cfg["seq_length"], n_targets),
                                           chunks=True,
                                           dtype=np.float32,
                                           compression='gzip')
        q_stds = out_f.create_dataset('q_stds',
                                      shape=(0, 1),
                                      maxshape=(None, 1),
                                      dtype=np.float32,
                                      compression='gzip',
                                      chunks=True)
        sample_2_basin = out_f.create_dataset('sample_2_basin',
                                              shape=(0,),
                                              maxshape=(None,),
                                              dtype="S11",
                                              compression='gzip',
                                              chunks=True)

        scalers = {
            'dyn_mean': np.zeros(n_dyn_inputs),
            'dyn_std': np.zeros(n_dyn_inputs),
            'target_mean': np.zeros(n_targets),
            'target_std': np.zeros(n_targets)
        }
        total_samples = 0

        basins_without_train_data = []

        for basin in tqdm(basins, file=sys.stdout):

            if additional_features:
                add_features = [d[basin] for d in additional_features]
            else:
                add_features = []

            try:
                dataset = get_basin_dataset(basin=basin, cfg=cfg, mode="train", additional_features=add_features)
            except NoTrainDataError as error:
                # skip basin
                basins_without_train_data.append(basin)
                continue

            num_samples = len(dataset)
            total_samples = dyn_input_data.shape[0] + num_samples

            basin_scaler = dataset.get_scaler()

            scalers["dyn_mean"] += num_samples * basin_scaler["dyn_mean"]
            scalers["dyn_std"] += num_samples * basin_scaler["dyn_std"]
            scalers["target_mean"] += num_samples * basin_scaler["target_mean"]
            scalers["target_std"] += num_samples * basin_scaler["target_std"]

            # store input and output samples
            dyn_input_data.resize((total_samples, cfg["seq_length"], n_dyn_inputs))
            dyn_input_data[-num_samples:, :, :] = dataset.x_d.numpy()

            target_data.resize((total_samples, cfg["seq_length"], n_targets))
            target_data[-num_samples:, :, :] = dataset.y.numpy()

            if n_stat > 0:
                x_stat = dataset.x_s.numpy()
                stat_input_data.resize((total_samples, n_stat))
                # the non-CAMELS stat features are stored at the end of the combined features
                stat_input_data[-num_samples:, :] = x_stat[:, -n_stat:]

            # additionally store std of discharge of this basin for each sample
            q_stds.resize((total_samples, 1))
            q_std_array = np.array([dataset.q_std] * num_samples, dtype=np.float32).reshape(-1, 1)
            q_stds[-num_samples:, :] = q_std_array

            sample_2_basin.resize((total_samples,))
            str_arr = np.array([basin.encode("ascii", "ignore")] * num_samples)
            sample_2_basin[-num_samples:] = str_arr

            out_f.flush()

    if basins_without_train_data:
        print("### The following basins were skipped, since they don't have discharge observations in the train period")
        print(basins_without_train_data)

    for key in scalers:
        scalers[key] /= total_samples

    if n_stat > 0:
        with h5py.File(h5_file, 'r') as f:
            scalers["stat_mean"] = f["static_inputs"][:].mean(axis=0)
            scalers["stat_std"] = f["static_inputs"][:].std(axis=0)

    if cfg.get("camels_attributes", []):
        attr_means, attr_stds = get_camels_scaler(data_dir=cfg["data_dir"],
                                                  basins=basins,
                                                  attributes=cfg["camels_attributes"])
        scalers["camels_attr_mean"] = attr_means
        scalers["camels_attr_std"] = attr_stds

    # sanity check that no std for any feature is 0, which results in NaN values during training
    problems_in_feature_std = []
    for k, v in scalers.items():
        # skip attributes, which were already tested above
        if k.endswith('_std') and ('attr' not in k):
            if any(v == 0) or any(np.isnan(v)):
                problems_in_feature_std.append((k, list(np.argwhere(np.isnan(v) | (v == 0)).flatten())))
    if problems_in_feature_std:
        print("### ERROR: Zero or NaN std encountered in the following features.")
        for k, pos in problems_in_feature_std:
            print(f"In scaler for {k} at position {pos}")
        raise RuntimeError

    with scaler_file.open("wb") as fp:
        pickle.dump(scalers, fp)

    # already normalize all data, so we don't have to do this while training
    with h5py.File(h5_file, 'r+') as f:
        print(f"Applying normalization in chunks of {CHUNK_SIZE} using global statistics")
        # perform iteration in chunks, for allowing to run on low memory systems

        n_batches = f["dynamic_inputs"].shape[0] // CHUNK_SIZE
        if f["dynamic_inputs"].shape[0] % CHUNK_SIZE > 0:
            n_batches += 1

        for i in tqdm(range(n_batches), file=sys.stdout):

            start_idx = i * CHUNK_SIZE
            end_idx = (i + 1) * CHUNK_SIZE
            if end_idx > f["dynamic_inputs"].shape[0]:
                slice_obj = slice(start_idx, None)
            else:
                slice_obj = slice(start_idx, end_idx)

            data = f["dynamic_inputs"]
            data[slice_obj] = (data[slice_obj] - scalers["dyn_mean"]) / scalers["dyn_std"]

            data = f["target_data"]
            if cfg.get("zero_center_target", True):
                data[slice_obj] = (data[slice_obj] - scalers["target_mean"]) / scalers["target_std"]
            else:
                data[slice_obj] = data[slice_obj] / scalers["target_std"]

            if n_stat > 0:
                data = f["static_inputs"]
                data[slice_obj] = (data[slice_obj] - scalers["stat_mean"]) / scalers["stat_std"]

            f.flush()
