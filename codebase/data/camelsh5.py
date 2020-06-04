"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from codebase.data.basedataseth5 import BaseDatasetH5
from codebase.data.utils import load_camels_attributes


class CamelsH5(BaseDatasetH5):

    def __init__(self, cfg: dict):
        super(CamelsH5, self).__init__(cfg)

    def _load_attributes(self):
        if self.cfg.get("camels_attributes", []):
            df = load_camels_attributes(data_dir=self.cfg["data_dir"], basins=self.basins)

            drop_cols = [c for c in df.columns if c not in self.cfg["camels_attributes"]]

            df = df.drop(drop_cols, axis=1)

            if not all([col in list(self.scaler["camels_attr_mean"].index) for col in df.columns]):
                raise RuntimeError("Camels attributes in scaler file do not match the config file.")

            # normalize data
            df = (df - self.scaler['camels_attr_mean']) / self.scaler['camels_attr_std']

            # fix the order of the columns to be alphabetically
            df = df.sort_index(axis=1)
        else:
            df = None

        return df
