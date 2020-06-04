"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import matplotlib.pyplot as plt
import numpy as np


def percentile_plot(y, y_hat, title: str = ''):
    fig, ax = plt.subplots()

    y_median = np.median(y_hat, axis=-1).flatten()
    y_25 = np.percentile(y_hat, 25, axis=-1).flatten()
    y_75 = np.percentile(y_hat, 75, axis=-1).flatten()
    y_10 = np.percentile(y_hat, 10, axis=-1).flatten()
    y_90 = np.percentile(y_hat, 90, axis=-1).flatten()
    y_05 = np.percentile(y_hat, 5, axis=-1).flatten()
    y_95 = np.percentile(y_hat, 95, axis=-1).flatten()

    x = np.arange(len(y_05))
    ax.fill_between(x, y_05, y_95, color='#edf8b1', label='5-95 percentile')
    ax.fill_between(x, y_10, y_90, color='#7fcdbb', label='10-90 percentile')
    ax.fill_between(x, y_25, y_75, color="#2c7fb8", label='25-75 percentile')
    ax.plot(y_median, '-', color='red', label="median")
    ax.plot(y.flatten(), '--', color='black', label="observed")
    ax.legend()
    ax.set_title(title)

    return fig, ax


def regression_plot(y, y_hat, title: str = ''):
    fig, ax = plt.subplots()

    ax.plot(y.flatten(), label="observed")
    ax.plot(y_hat.flatten(), label="simulated")
    ax.legend()
    ax.set_title(title)

    return fig, ax


def uncertainty_plot(y, y_hat, title: str = ''):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7, 3), gridspec_kw={'width_ratios': [3, 5]})

    y_flat = y.flatten()
    y_len = len(y_flat)

    # only take part of y to have a better zoom-in
    bnd = round(y_len / 4)
    x_bnd = np.arange(bnd)

    # hydrograph:
    y_r = [0, 0, 0, 0, 0]  # used later for calibration-plot
    quantiles = [0.9, 0.80, 0.50, 0.20, 0.1]
    labels_and_colors = {
        'labels': [
            '5-95 percentile', '10-90 percentile', '25-75 percentile', '40-60 percentile',
            '45-55 percentile'
        ],
        'colors': ['#FDE725', '#35B779', '#31688E', '#440154', '#440154']
    }
    for idx in range(5):
        lb = round(50 - (quantiles[idx] * 100) / 2)
        ub = round(50 + (quantiles[idx] * 100) / 2)
        y_lb = np.percentile(y_hat, lb, axis=-1).flatten()
        y_ub = np.percentile(y_hat, ub, axis=-1).flatten()
        y_r[idx] = np.sum(((y_flat > y_lb) * (y_flat < y_ub))) / y_len
        if idx <= 2:
            axs[1].fill_between(x_bnd,
                                y_lb[0:bnd],
                                y_ub[0:bnd],
                                color=labels_and_colors['colors'][idx],
                                label=labels_and_colors['labels'][idx])

    y_median = np.median(y_hat, axis=-1).flatten()
    axs[1].plot(y_median[0:bnd], '-', color='red', label="median")
    axs[1].plot(y_flat[0:bnd], '--', color='black', label="observed")
    axs[1].legend()

    # calibration-plot:
    axs[0].plot([0, 1], [0, 1], 'k--')
    for idx in range(4):
        # move description out of the way:
        is_quantile_small = quantiles[idx] <= 0.5
        ha_argument = 'right' if is_quantile_small else 'left'
        text_pos = 1 if is_quantile_small else 0
        l_coord = [text_pos, quantiles[idx]] if is_quantile_small else [quantiles[idx], text_pos]

        axs[0].plot(l_coord, [y_r[idx], y_r[idx]], ':', color='#ffb95a')
        axs[0].text(text_pos,
                    y_r[idx],
                    f'{round(y_r[idx], 2)}',
                    fontsize=8,
                    va='center',
                    ha=ha_argument,
                    c='#ffb95a',
                    backgroundcolor='w')

    axs[0].plot(quantiles, y_r, 'ro-')
    axs[0].set_axisbelow(True)
    axs[0].yaxis.grid(color='#ECECEC', linestyle='dashed')
    axs[0].xaxis.grid(color='#ECECEC', linestyle='dashed')
    axs[0].xaxis.set_ticks(np.arange(0, 1, 0.2))
    axs[0].yaxis.set_ticks(np.arange(0, 1, 0.2))
    axs[0].set_xlabel("quantiles")
    axs[0].set_ylabel("obs in quantiles")
    axs[0].set_title(title)

    fig.tight_layout()

    return fig, axs
