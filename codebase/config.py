"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from collections import OrderedDict
from pathlib import Path, PosixPath
from typing import Dict

import pandas as pd
from ruamel.yaml import YAML


def read_config(cfg_path: PosixPath) -> Dict:
    if cfg_path.exists():
        with cfg_path.open('r') as fp:
            yaml = YAML(typ="safe")
            cfg = yaml.load(fp)
    else:
        raise FileNotFoundError(cfg_path)

    cfg = parse_config(cfg)

    return cfg


def dump_config(cfg: Dict, folder: PosixPath, filename: str = 'config.yml'):
    cfg_path = folder / filename
    if not cfg_path.exists():
        with cfg_path.open('w') as fp:
            temp_cfg = {}
            for key, val in cfg.items():
                if any([x in key for x in ['dir', 'path', 'file']]):
                    if isinstance(val, list):
                        temp_list = []
                        for elem in val:
                            temp_list.append(str(elem))
                        temp_cfg[key] = temp_list
                    else:
                        temp_cfg[key] = str(val)
                elif key.endswith('_date'):
                    if isinstance(val, list):
                        temp_list = []
                        for elem in val:
                            temp_list.append(elem.strftime(format="%d/%m/%Y"))
                        temp_cfg[key] = temp_list
                    else:
                        temp_cfg[key] = val.strftime(format="%d/%m/%Y")
                else:
                    temp_cfg[key] = val

            yaml = YAML()
            yaml.dump(dict(OrderedDict(sorted(temp_cfg.items()))), fp)
    else:
        FileExistsError(cfg_path)


def parse_config(cfg: Dict) -> Dict:

    for key, val in cfg.items():
        # convert all path strings to PosixPath objects
        if any([x in key for x in ['dir', 'path', 'file']]):
            if (val is not None) and (val != "None"):
                if isinstance(val, list):
                    temp_list = []
                    for element in val:
                        temp_list.append(Path(element))
                    cfg[key] = temp_list
                else:
                    cfg[key] = Path(val)
            else:
                cfg[key] = None

        # convert Dates to pandas Datetime indexs
        elif key.endswith('_date'):
            if isinstance(val, list):
                temp_list = []
                for elem in val:
                    temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                cfg[key] = temp_list
            else:
                cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')

        elif any(key == x for x in ["static_inputs", "camels_attributes", "hydroatlas_attributes"]):
            if val is None:
                cfg[key] = []
        else:
            pass

    # Add more config parsing if necessary

    return cfg


if __name__ == "__main__":
    config = read_config(
        Path(
            "/home/frederik/Remote/publicwork/kratzert/projects/lstm_based_hydrology/codebase/utils/generated_configs/config_1.yml"
        ))
