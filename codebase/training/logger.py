"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import subprocess
import numpy as np
import torch

from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from codebase.config import dump_config
from pathlib import Path


class Logger(object):

    def __init__(self, cfg: dict):
        self._train = True
        self.log_interval = int(cfg['log_interval'])
        self.checkpoint = int(cfg['save_weights_every'])
        self.log_dir = cfg['run_dir']

        if cfg.get('log_n_figures', 0) > 0:
            self.trainlog_dir = cfg['trainlog_dir']

        # dump configuration
        try:
            git_output = subprocess.check_output(["git", "describe", "--always"])
            cfg['commit_hash'] = git_output.strip().decode('ascii')
        except subprocess.CalledProcessError:
            cfg["commit_hash"] = ''
        cfg['log_dir'] = self.log_dir
        dump_config(cfg, folder=self.log_dir)

        self.epoch = 0
        self.update = 0
        self._metrics = defaultdict(list)
        self.writer = None

    @property
    def metrics(self):
        return self._metrics

    @property
    def tag(self):
        return "train" if self._train else "valid"

    def train(self):
        """ Log on the training data. """
        self._train = True
        return self

    def valid(self):
        """ Log on the validation data. """
        self._train = False
        return self

    def start_tb(self):
        """ Start tensorboard logging. """
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def stop_tb(self):
        """ Stop tensorboard logging. """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def log_figures(self, figures, preamble=""):
        if self.writer is not None:
            self.writer.add_figure('validation/timeseries', figures, global_step=self.epoch)

        for idx, figure in enumerate(figures):
            figure.savefig(Path(self.trainlog_dir, preamble + f'_epoch{self.epoch}_{idx + 1}'), dpi=300)

    def log_step(self, **kwargs):
        """ Log the results of a single step within an epoch. """
        for k, v in kwargs.items():
            self._metrics[k].append(v)

        if not self._train:
            return

        self.update += 1

        if self.log_interval <= 0 or self.writer is None:
            return

        if self.update % self.log_interval == 0:
            tag = self.tag
            for k, v in kwargs.items():
                self.writer.add_scalar('/'.join([tag, k]), v, self.update)

    def summarise(self, model: torch.nn.Module = None):
        """ Log the results of the entire epoch. """
        # summarize statistics of training epoch
        if self._train:
            self.epoch += 1

            # summarize training
            avg_loss = np.nanmean(self._metrics["loss"]) if self._metrics["loss"] else np.nan

            if self.writer is not None:
                self.writer.add_scalar('/'.join([self.tag, 'avg_loss']), avg_loss, self.epoch)

        # summarize validation
        else:
            for k, v in self._metrics.items():
                means = np.nanmean(v) if v else np.nan
                medians = np.nanmedian(v) if v else np.nan
                self.writer.add_scalar('/'.join([self.tag, f'mean_{k.lower()}']), means, self.epoch)
                self.writer.add_scalar('/'.join([self.tag, f'median_{k.lower()}']), medians, self.epoch)

        # store model checkpoint
        if not self._train or model is None or self.checkpoint <= 0:
            pass
        elif self.epoch % self.checkpoint == 0:
            weight_path = self.log_dir / f"model_epoch{self.epoch:03d}.pt"
            torch.save(model.state_dict(), str(weight_path))

        # return value for print in console
        if self._train:
            value = avg_loss
        else:
            if 'NSE' in self._metrics.keys():
                value = np.median(self._metrics["NSE"])
            else:
                value = None

        # clear buffer
        self._metrics = defaultdict(list)

        return value
