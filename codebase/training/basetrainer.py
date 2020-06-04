"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import pickle
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from codebase.data import get_h5_dataset
from codebase.data.hdf5utils import create_h5_file
from codebase.data.utils import load_basin_file
from codebase.evaluation.tester import Tester
from codebase.modelzoo import get_model
from codebase.training import get_loss_obj, get_optimizer
from codebase.training.logger import Logger


class BaseTrainer(object):

    def __init__(self, cfg: dict):
        super(BaseTrainer, self).__init__()
        self._cfg = cfg
        self.model = None
        self.optimizer = None
        self.loss_obj = None
        self.logger = None
        self.loader = None
        # load train basin list and add number of basins to the config
        self.basins = load_basin_file(cfg["train_basin_file"])
        self._cfg["number_of_basins"] = len(self.basins)

        self._set_random_seeds()
        self._set_device()
        self._create_folder_structure()
        self._prepare_train_data()

        if (cfg["validate_every"] is not None) and (cfg["validate_every"] > 0):
            self.validator = Tester(cfg=self._cfg, run_dir=self._cfg["run_dir"], mode="validation", init_model=False)
        else:
            self.validator = None

    @property
    def cfg(self):
        return self._cfg

    def initialize_training(self):
        self.model = get_model(cfg=self.cfg).to(self.device)
        self.optimizer = get_optimizer(model=self.model, cfg=self.cfg)
        self.loss_obj = get_loss_obj(cfg=self.cfg)

        ds = get_h5_dataset(cfg=self.cfg)
        self.loader = DataLoader(ds,
                                 batch_size=self.cfg["batch_size"],
                                 shuffle=True,
                                 num_workers=self.cfg["num_workers"])

        self.logger = Logger(cfg=self.cfg)
        if self.cfg["log_tensorboard"]:
            self.logger.start_tb()

    def train_and_validate(self):
        for epoch in range(1, self.cfg["epochs"] + 1):
            # set new learning rate
            if ((self.cfg["learning_rate"] is not None) and (epoch in self.cfg["learning_rate"].keys())):
                print(f"Setting learning rate to {self.cfg['learning_rate'][epoch]}")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.cfg["learning_rate"][epoch]

            self._train_epoch(epoch=epoch)
            _ = self.logger.summarise(self.model)

            if (self.validator is not None) and (epoch % self.cfg["validate_every"] == 0):
                self.validator.evaluate(epoch=epoch,
                                        save_results=self.cfg['save_validation_results'],
                                        metrics=self.cfg.get('metrics', []),
                                        model=self.model,
                                        logger=self.logger.valid())

                valid_nse = self.logger.summarise(self.model)
                if valid_nse is not None:
                    print(f" -- Median validation NSE: {valid_nse}")

    def _train_epoch(self, **kwargs):
        raise NotImplementedError

    def _set_random_seeds(self):
        if self.cfg["seed"] is None:
            self.cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

        # fix random seeds for various packages
        random.seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        torch.cuda.manual_seed(self.cfg["seed"])
        torch.manual_seed(self.cfg["seed"])

    def _set_device(self):
        if self.cfg["device"] is not None:
            if "cuda" in self.cfg["device"]:
                gpu_id = int(self.cfg["device"].split(':')[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
                else:
                    self.device = torch.device(self.cfg["device"])
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"### Device {self.device} will be used for training")

    def _create_folder_structure(self):
        now = datetime.now()
        day = f"{now.day}".zfill(2)
        month = f"{now.month}".zfill(2)
        hour = f"{now.hour}".zfill(2)
        minute = f"{now.minute}".zfill(2)
        if self.cfg["experiment_name"] is not None:
            experiment_name = self.cfg["experiment_name"]
        else:
            experiment_name = "run"
        run_name = f'{experiment_name}_{day}{month}_{hour}{minute}'
        code_root = Path(__file__).absolute().parent.parent.parent
        if ('run_dir' in self.cfg.keys()) and (self.cfg["run_dir"] is not None):
            self.cfg["run_dir"] = self.cfg["run_dir"] / run_name
        else:
            self.cfg['run_dir'] = code_root / "runs" / run_name
        if not self.cfg["run_dir"].is_dir():
            self.cfg["train_dir"] = self.cfg["run_dir"] / "train_data"
            self.cfg['train_dir'].mkdir(parents=True)
        else:
            raise RuntimeError(f"There is already a folder at {self.cfg['run_dir']}")
        if self.cfg.get('log_n_figures', 0) > 0:
            self.cfg["img_dir"] = Path(self.cfg["run_dir"], "img_log")
            self.cfg["trainlog_dir"] = Path(self.cfg['img_dir'], 'progress_log')
            self.cfg["trainlog_dir"].mkdir(parents=True)

        print(f"### Folder structure created at {self.cfg['run_dir']}")

    def _prepare_train_data(self):
        if (self.cfg["h5_file"] is None) and (self.cfg["scaler_file"] is None):
            self._create_train_data()
        elif (self.cfg["scaler_file"] is not None) and (self.cfg["h5_file"] is None):
            raise RuntimeError("If scaler file is defined, also h5_file has to be specified.")
        else:
            dst = self.cfg["train_dir"] / self.cfg["scaler_file"].name
            shutil.copyfile(self.cfg["scaler_file"], dst)

    def _create_train_data(self):
        self.cfg["h5_file"] = self.cfg["train_dir"] / 'train_data.h5'
        self.cfg["scaler_file"] = self.cfg["train_dir"] / "train_data_scaler.p"

        additional_features = []
        if self.cfg["additional_feature_files"]:
            for file in self.cfg["additional_feature_files"]:
                with open(file, "rb") as fp:
                    additional_features.append(pickle.load(fp))

        if self.cfg["dataset"] in ["camels_us"]:
            create_h5_file(basins=self.basins,
                           cfg=self.cfg,
                           h5_file=self.cfg["h5_file"],
                           scaler_file=self.cfg["scaler_file"],
                           additional_features=additional_features)
        else:
            raise ValueError(f"Unknown dataset type {self.cfg['dataset']}")
