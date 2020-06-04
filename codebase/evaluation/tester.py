"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import random
import pickle
import sys
from collections import defaultdict
from pathlib import PosixPath

import numpy as np
import pandas as pd
import torch
import xarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from codebase.data import get_basin_dataset
from codebase.data.utils import load_basin_file
from codebase.evaluation import plots
from codebase.evaluation.metrics import calculate_metrics
from codebase.modelzoo import get_model
from codebase.training.logger import Logger


class Tester(object):

    def __init__(self, cfg: dict, run_dir: PosixPath, mode: str = "test", init_model: bool = True):
        self.cfg = cfg
        self.run_dir = run_dir
        self.init_model = init_model
        if mode in ["train", "validation", "test"]:
            self.mode = mode
        else:
            raise ValueError(f'Invalid mode {mode}. Must be one of ["train", "validation", "test"]')

        # determine device
        self._set_device()

        if self.init_model:
            self.model = get_model(cfg).to(self.device)

        # pre-initialize variables, defined in class methods
        self.basins = None
        self.scaler = None
        self.id_to_int = {}
        self.additional_features = []

        self._load_run_data()

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

    def _load_run_data(self):
        """Load run specific data from run directory"""

        # get list of basins
        self.basins = load_basin_file(self.cfg[f"{self.mode}_basin_file"])

        # load feature scaler
        scaler_file = self.run_dir / 'train_data' / self.cfg["scaler_file"].name
        with scaler_file.open('rb') as fp:
            self.scaler = pickle.load(fp)

        # load basin_id to integer dictionary for one-hot-encoding
        if self.cfg["use_basin_id_encoding"]:
            file_path = self.run_dir / "train_data" / "id_to_int.p"
            with file_path.open("rb") as fp:
                self.id_to_int = pickle.load(fp)

    def _get_weight_file(self, epoch: int):
        """Get file path to weight file"""
        if epoch is None:
            weight_file = sorted(list(self.run_dir.glob('*.pt')))[-1]
        else:
            weight_file = self.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"

        return weight_file

    def _load_weights(self, epoch: int = None):
        """Load weights of a certain (or the last) epoch into the model."""
        weight_file = self._get_weight_file(epoch)

        print(f"Using the model weights from {weight_file}")
        self.model.load_state_dict(torch.load(weight_file, map_location=self.device))

    def evaluate(self,
                 epoch: int = None,
                 save_results: bool = True,
                 metrics: list = [],
                 model: torch.nn.Module = None,
                 logger: Logger = None) -> dict:
        """Evaluate the model.
        
        Parameters
        ----------
        epoch : int
            (optional) Define a specific epoch to evaluate. By default, the weights of the last 
            epoch are used. 
        save_results : bool
            (optional) If True, stores the evaluation results in the run directory. By default, 
            True.
        metrics: list
            (optional) List of metrics to compute during evaluation. Still WIP.
        model: torch.nn.Module
            (optional) If a model is passed, this is used for validation.
        logger: Logger
            (optional) Logger can be passed during training to log metrics

        Returns
        -------
            A dictionary containing per basin one xarray with the evaluation results.
        """
        if model is None:
            if self.init_model:
                self._load_weights(epoch=epoch)
                model = self.model
            else:
                raise RuntimeError("No model was initialized for the evaluation")

        # during validation, depending on settings, only evaluate on a random subset of basins
        basins = self.basins
        if self.mode == "validation":
            if (self.cfg["validate_n_random_basins"] is not None) and (len(basins) >
                                                                       self.cfg["validate_n_random_basins"]):
                random.shuffle(basins)
                basins = basins[:self.cfg["validate_n_random_basins"]]

        model.eval()
        results = defaultdict(dict)

        pbar = tqdm(basins, file=sys.stdout)
        pbar.set_description('# Validation' if self.mode == "validation" else "# Evaluation")

        for basin in pbar:

            # check for additional features
            if self.additional_features:
                additional_features = [f[basin] for f in additional_features]
            else:
                additional_features = []

            ds = get_basin_dataset(basin=basin,
                                   cfg=self.cfg,
                                   mode=self.mode,
                                   additional_features=additional_features,
                                   id_to_int=self.id_to_int,
                                   scaler=self.scaler)

            loader = DataLoader(ds, batch_size=self.cfg["batch_size"], num_workers=0)

            if self.cfg["head"] == "regression":
                y_hat, y = self._evaluate_regression(model, loader)
            else:
                msg = f"No evaluation method implemented for {self.cfg['head']} head"
                raise NotImplementedError(msg)

            # rescale predictions
            y_hat = y_hat * self.scaler["target_std"]
            if self.cfg.get("zero_center_target", True):
                y_hat = y_hat + self.scaler["target_mean"]

            # create xarray
            data = {}
            if self.cfg["head"] == "regression":
                for i, var in enumerate(self.cfg["target_variable"]):
                    data[f"{var}_obs"] = (('date', 'time_step'), y[:, :, i])
                    data[f"{var}_sim"] = (('date', 'time_step'), y_hat[:, :, i])

            if ds.period_start + pd.DateOffset(days=self.cfg["seq_length"] - 1) > self.cfg[f"{self.mode}_start_date"]:
                start_date = ds.period_start + pd.DateOffset(days=self.cfg["seq_length"] - 1)
            else:
                start_date = self.cfg[f"{self.mode}_start_date"]

            # determine the end of the first sequence (first target in sequence-to-one)
            date_range = pd.date_range(start=start_date, end=self.cfg[f"{self.mode}_end_date"])

            xr = xarray.Dataset(data_vars=data,
                                coords={
                                    'date': date_range,
                                    'time_step': np.arange(-self.cfg["predict_last_n"] + 1, 1)
                                })
            results[basin]['xr'] = xr

            if metrics:
                qobs_variable_name = [col for col in self.cfg["target_variable"] if "qobs" in col.lower()]
                if qobs_variable_name:
                    qobs_variable_name = qobs_variable_name[0]

                    # check if not empty (in case no streamflow data exist in validation period
                    qobs = xr[f"{qobs_variable_name}_obs"].sel(dict(time_step=0))
                    qsim = xr[f"{qobs_variable_name}_sim"].sel(dict(time_step=0))
                    if (len(qsim.shape) > 1) and (qsim.shape[-1] > 0):
                        # print("Using the mean discharge for metric calculation")
                        qsim = qsim.mean(axis=-1)
                    values = calculate_metrics(qobs, qsim, metrics=metrics)
                    if logger is not None:
                        logger.log_step(**values)
                    for k, v in values.items():
                        results[basin][k] = v

        if (self.mode == "validation") and (self.cfg.get("log_n_figures", 0) > 0):
            self._create_and_log_figures(results, logger, epoch)

        if save_results:
            self._save_results(results, epoch)

        return results

    def _create_and_log_figures(self, results, logger, epoch):
        basins = list(results.keys())
        random.shuffle(basins)
        figures = []
        qobs_variable_name = [col for col in self.cfg["target_variable"] if "qobs" in col.lower()]
        if qobs_variable_name:
            qobs_variable_name = qobs_variable_name[0]
            max_figures = min(self.cfg["validate_n_random_basins"], self.cfg["log_n_figures"])
            for i in range(max_figures):
                xr = results[basins[i]]['xr']
                qobs = xr[f"{qobs_variable_name}_obs"].values
                qsim = xr[f"{qobs_variable_name}_sim"].values
                fig, _ = plots.regression_plot(qobs, qsim, title=f"Basin {basins[i]} - Epoch {epoch}")
                figures.append(fig)
            logger.log_figures(figures, preamble="hydrograph")
        else:
            print("Didn't find qobs variable in target (Name must contain 'qobs'")

    def _save_results(self, results: dict, epoch: int = None):
        # use name of weight file as part of the result folder name
        weight_file = self._get_weight_file(epoch=epoch)

        result_file = self.run_dir / self.mode / weight_file.stem / f"{self.mode}_results.p"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        with result_file.open("wb") as fp:
            pickle.dump(results, fp)

        print(f"Stored results at {result_file}")

    def _evaluate_regression(self, model, loader):
        """Evaluate regression model"""
        preds, obs = None, None

        with torch.no_grad():
            for x_d, x_s, x_one_hot, y in loader:

                x_d, x_s = x_d.to(self.device), x_s.to(self.device)
                x_one_hot = x_one_hot.to(self.device)
                y_hat = model(x_d, x_s, x_one_hot)[0]

                y_sub = y[:, -self.cfg["predict_last_n"]:, :]
                y_hat_sub = y_hat[:, -self.cfg["predict_last_n"]:, :]

                if preds is None:
                    preds = y_hat_sub.detach().cpu()
                    obs = y_sub
                else:
                    preds = torch.cat((preds, y_hat_sub.detach().cpu()), 0)
                    obs = torch.cat((obs, y_sub.detach().cpu()), 0)

            preds = preds.numpy()
            obs = obs.numpy()

        return preds, obs
