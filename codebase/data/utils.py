"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from pathlib import Path, PosixPath
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import njit


def load_camels_attributes(data_dir: PosixPath, basins: List = []) -> pd.DataFrame:
    attributes_path = Path(data_dir) / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)
    # convert huc column to double digit strings
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if basins:
        # drop rows of basins not contained in the passed list
        drop_basins = [b for b in df.index if b not in basins]
        df = df.drop(drop_basins, axis=0)

    return df

@njit
def reshape_data(x_d: np.ndarray, y: np.ndarray, seq_length: int,
                 x_s: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray,]:

    num_samples, num_features = x_d.shape
    num_targets = y.shape[-1]

    x_d_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, seq_length, num_targets))

    if x_s is not None:
        x_s_new = np.zeros((num_samples - seq_length + 1, x_s.shape[-1]))
    else:
        x_s_new = None

    for i in range(0, x_d_new.shape[0]):
        x_d_new[i, :, :] = x_d[i:i + seq_length, :]
        y_new[i, :, :] = y[i:i + seq_length, :]
        if x_s is not None:
            x_s_new[i, :] = x_s[i + seq_length - 1, :]

    return x_d_new, x_s_new, y_new


def load_forcings(data_dir: PosixPath, basin: str, forcings: str) -> Tuple[pd.DataFrame, int]:
    forcing_path = data_dir / 'basin_mean_forcing' / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob('**/*_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    df = pd.read_csv(file_path, sep='\s+', header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # load area from header
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area


def load_discharge(data_dir: PosixPath, basin: str, area: int) -> pd.Series:

    discharge_path = data_dir / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # normalize discharge from cubic feed per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs


def get_camels_scaler(data_dir: PosixPath, basins: List, attributes: List):
    df = load_camels_attributes(data_dir=data_dir, basins=basins)
    drop_cols = [c for c in df.columns if c not in attributes]
    df = df.drop(drop_cols, axis=1)
    return df.mean(), df.std()


def load_basin_file(basin_file: PosixPath) -> List:
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = [basin.strip() for basin in basins]
    return basins


def attributes_sanity_check(data_dir: PosixPath, dataset: str, basins: list, attribute_list: list):
    if dataset == "camels_us":
        df = load_camels_attributes(data_dir, basins)
    drop_cols = [c for c in df.columns if c not in attribute_list]
    df = df.drop(drop_cols, axis=1)
    attributes = []
    if any(df.std() == 0.0) or any(df.std().isnull()):
        for k, v in df.std().iteritems():
            if (v == 0) or (np.isnan(v)):
                attributes.append(k)
    if attributes:
        msg = [
            "The following attributes have a std of zero or NaN, which results in NaN's ",
            "when normalizing the features. Remove the attributes from the attribute feature list ",
            "and restart the run. \n", f"Attributes: {attributes}"
        ]
        raise RuntimeError("".join(msg))
