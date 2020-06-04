"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from codebase.data.camelstxt import CamelsTXT
from codebase.data.camelsh5 import CamelsH5


def get_basin_dataset(basin: str, cfg: dict, mode: str, **kwargs):
    if cfg["dataset"] == 'camels_us':
        Dataset = CamelsTXT
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg['dataset']}")

    ds = Dataset(basin=basin,
                 cfg=cfg,
                 mode=mode,
                 additional_features=kwargs.get('additional_features', []),
                 id_to_int=kwargs.get('id_to_int', {}),
                 scaler=kwargs.get('scaler', {}))
    return ds


def get_h5_dataset(cfg: dict):
    if cfg["dataset"] == 'camels_us':
        Dataset = CamelsH5
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg['dataset']}")

    return Dataset(cfg)
