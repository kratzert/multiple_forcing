"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
from numba import njit
from xarray.core.dataarray import DataArray


def get_available_signatures() -> list:
    """Get list of available signatures."""
    signatures = [
        "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq", "q95", "q5",
        "q_mean", "hfd_mean", "baseflow_index", "slope_fdc", "stream_elas", "runoff_ratio"
    ]
    return signatures


def calculate_all_signatures(da: DataArray, **kwargs) -> dict:
    """Calculate all metrics with default values."""
    results = {
        "high_q_freq": high_q_freq(da),
        "high_q_dur": high_q_dur(da),
        "low_q_freq": low_q_freq(da),
        "low_q_dur": low_q_dur(da),
        "zero_q_freq": zero_q_freq(da),
        "q95": q95(da),
        "q5": q5(da),
        "q_mean": q_mean(da),
        "hfd_mean": hfd_mean(da),
        "baseflow_index": baseflow_index(da)[0],
        "slope_fdc": slope_fdc(da),
        "stream_elas": stream_elas(da, kwargs.get('prcp')),
        "runoff_ratio": runoff_ratio(da, kwargs.get('prcp'))
    }
    return results


def calculated_signatures(da: DataArray, signatures: list, **kwargs) -> dict:
    values = {}
    for signature in signatures:
        if signature == "high_q_freq":
            values["high_q_freq"] = high_q_freq(da)
        elif signature == "high_q_dur":
            values["high_q_dur"] = high_q_dur(da)
        elif signature == "low_q_freq":
            values["low_q_freq"] = low_q_freq(da)
        elif signature == "low_q_dur":
            values["low_q_dur"] = low_q_dur(da)
        elif signature == "zero_q_freq":
            values["zero_q_freq"] = zero_q_freq(da)
        elif signature == "q95":
            values["q95"] = q95(da)
        elif signature == "q5":
            values["q5"] = q5(da)
        elif signature == "q_mean":
            values["q_mean"] = q_mean(da)
        elif signature == "hfd_mean":
            values["hfd_mean"] = hfd_mean(da)
        elif signature == "baseflow_index":
            values["baseflow_index"] = baseflow_index(da)[0]
        elif signature == "slope_fdc":
            values["slope_fdc"] = slope_fdc(da)
        elif signature == "runoff_ratio":
            values["runoff_ratio"] = runoff_ratio(da, kwargs.get('prcp'))
        elif signature == "stream_elas":
            values["stream_elas"] = stream_elas(da, kwargs.get('prcp'))
        else:
            ValueError(f"Unknown signatures {signature}")


@njit
def _split_list(alist: list) -> list:
    newlist = []
    start = 0
    end = 0
    for index, value in enumerate(alist):
        if index < len(alist) - 1:
            if alist[index + 1] > value + 1:
                end = index + 1
                newlist.append(alist[start:end])
                start = end
        else:
            newlist.append(alist[start:len(alist)])
    return newlist


def high_q_dur(da: DataArray, threshold: float = 9.) -> float:
    median_flow = float(da.median())
    idx = np.where(da.values > threshold * median_flow)[0]
    if len(idx) > 0:
        periods = _split_list(idx)
        hqd = np.mean([len(p) for p in periods])
    else:
        hqd = np.nan
    return hqd


def low_q_dur(da: DataArray, threshold: float = 0.2) -> float:
    mean_flow = float(da.mean())
    idx = np.where(da.values < threshold * mean_flow)[0]
    if len(idx) > 0:
        periods = _split_list(idx)
        lqd = np.mean([len(p) for p in periods])
    else:
        lqd = np.nan
    return lqd


def zero_q_freq(da: DataArray) -> float:

    # number of days with zero flow
    n_days = (da == 0).sum()

    return float(n_days / len(da))


def high_q_freq(da: DataArray, coord: str = 'date', threshold: float = 9.) -> float:

    # determine the date of the first January 1st in the data period
    first_date = da.coords[coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date == datetime.strptime(f'{first_date.year}-01-01', '%Y-%m-%d'):
        start_date = first_date
    else:
        start_date = datetime.strptime(f'{first_date.year + 1}-01-01', '%Y-%m-%d')

    # end date of the first full year period
    end_date = start_date + relativedelta(years=1) - relativedelta(days=1)

    # determine the median flow over the entire period
    median_flow = da.median(skipna=True)

    hqfs = []
    while end_date < last_date:

        data = da.sel({coord: slice(start_date, end_date)})

        # number of days with discharge higher than threshold * median in a one year period
        n_days = (data > (threshold * median_flow)).sum()

        hqfs.append(float(n_days))

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.mean(hqfs)


def low_q_freq(da: DataArray, coord: str = 'date', threshold: float = 0.2) -> float:

    # determine the date of the first January 1st in the data period
    first_date = da.coords[coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date == datetime.strptime(f'{first_date.year}-01-01', '%Y-%m-%d'):
        start_date = first_date
    else:
        start_date = datetime.strptime(f'{first_date.year + 1}-01-01', '%Y-%m-%d')

    # end date of the first full year period
    end_date = start_date + relativedelta(years=1) - relativedelta(days=1)

    # determine the mean flow over the entire period
    mean_flow = da.mean(skipna=True)

    lqfs = []
    while end_date < last_date:

        data = da.sel({coord: slice(start_date, end_date)})

        # number of days with discharge lower than threshold * median in a one year period
        n_days = (data < (threshold * mean_flow)).sum()

        lqfs.append(float(n_days))

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.mean(lqfs)


def hfd_mean(da: DataArray, coord: str = 'date') -> float:

    # determine the date of the first October 1st in the data period
    first_date = da.coords[coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date > datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d'):
        start_date = datetime.strptime(f'{first_date.year + 1}-10-01', '%Y-%m-%d')
    else:
        start_date = datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d')

    end_date = start_date + relativedelta(years=1) - relativedelta(days=1)

    doys = []
    while end_date < last_date:

        # compute cumulative sum for the selected period
        data = da.sel({coord: slice(start_date, end_date)})
        cs = data.cumsum(skipna=True)

        # find days with more cumulative discharge than the half annual sum
        days = np.where(~np.isnan(cs.where(cs > data.sum(skipna=True) / 2).values))[0]

        # ignore days without discharge
        if len(days) > 0:
            # store the first day in the result array
            doys.append(days[0])

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.mean(doys)


def q5(da: DataArray) -> float:
    return float(da.quantile(0.05))


def q95(da: DataArray) -> float:
    return float(da.quantile(0.95))


def q_mean(da: DataArray) -> float:
    return float(da.mean())


@njit
def _baseflow_index_jit(streamflow: np.ndarray, alpha: float, warmup: int) -> (float, np.ndarray):
    # create buffer for the first (forward) pass for quickflow and streamflow
    quickflow1 = np.zeros_like(streamflow)
    baseflow1 = np.zeros_like(streamflow)

    # the first time step is hardcoded, quickflow == streamflow, baseflow = 0
    quickflow1[0] = streamflow[0]

    for i in range(1, len(streamflow)):
        quickflow1[i] = alpha * quickflow1[i - 1] + (1 + alpha) * (streamflow[i] -
                                                                   streamflow[i - 1]) / 2
        if quickflow1[i] > 0:
            baseflow1[i] = streamflow[i] - quickflow1[i]
        else:
            baseflow1[i] = streamflow[i]

    # create buffer for the second (backward) pass for quickflow and streamflow
    quickflow2 = np.zeros_like(streamflow)
    baseflow2 = np.zeros_like(streamflow)

    # in the second pass, the last time step is hardcoded, quickflow2 == baseflow1
    quickflow2[-1] = baseflow1[-1]
    for i in range(len(streamflow) - 2, -1, -1):
        quickflow2[i] = alpha * quickflow2[i + 1] + (1 + alpha) * (baseflow1[i] -
                                                                   baseflow1[i + 1]) / 2

        if quickflow2[i] > 0:
            baseflow2[i] = baseflow1[i] - quickflow2[i]
        else:
            baseflow2[i] = baseflow1[i]

    # create buffer for the third (forward) pass for quickflow and streamflow
    quickflow3 = np.zeros_like(streamflow)
    baseflow3 = np.zeros_like(streamflow)

    # the first time step is hardcoded, quickflow3 == baseflow2, baseflow3 = 0
    quickflow3[0] = baseflow2[0]
    for i in range(1, len(streamflow)):
        quickflow3[i] = alpha * quickflow3[i - 1] + (1 + alpha) * (baseflow2[i] -
                                                                   baseflow2[i - 1]) / 2
        if quickflow3[i] > 0:
            baseflow3[i] = baseflow2[i] - quickflow3[i]
        else:
            baseflow3[i] = baseflow2[i]

    bf_index = np.sum(baseflow3[warmup:-warmup]) / np.sum(streamflow[warmup:-warmup])

    return bf_index, baseflow3[warmup:-warmup]


def baseflow_index(da: DataArray, alpha: float = 0.98, warmup: int = 30) -> (float, DataArray):
    """Currently just implemented for daily flows (i.e. 3 passes, see Section 2.3 Landson et al. 2013"""
    # create numpy array from streamflow and add the mirrored discharge of length 'window' to the start and end
    streamflow = np.zeros((da.size + 2 * warmup))
    streamflow[warmup:-warmup] = da.values
    streamflow[:warmup] = da.values[1:warmup + 1][::-1]
    streamflow[-warmup:] = da.values[-warmup - 1:-1][::-1]

    # call jit compiled function to calculate baseflow
    bf_index, baseflow = _baseflow_index_jit(streamflow, alpha, warmup)

    # parse baseflow as a DataArray using the coordinates of the streamflow array
    da_baseflow = da.copy()
    da_baseflow.data = baseflow

    return bf_index, da_baseflow


def slope_fdc(da: DataArray, lower_quantile: float = 0.33, upper_quantile: float = 0.66) -> float:
    # sort discharge by descending order
    fdc = da.sortby(da, ascending=False)

    # get idx of lower and upper quantile
    idx_lower = np.round(lower_quantile * len(fdc)).astype(int)
    idx_upper = np.round(upper_quantile * len(fdc)).astype(int)

    value = (np.log(fdc[idx_lower].values + 1e-8)
            ) - np.log(fdc[idx_upper].values + 1e-8) / (upper_quantile - lower_quantile)

    return value


def runoff_ratio(da: DataArray, prcp: DataArray) -> float:
    # get precip coordinate name (to avoid problems with 'index' or 'date')
    coord_name = list(prcp.coords.keys())[0]

    # slice prcp to the same time window as the discharge
    prcp = prcp.sel({coord_name: slice(da.coords['date'][0], da.coords['date'][-1])})

    # calculate runoff ratio
    value = da.mean() / prcp.mean()

    return float(value)


def stream_elas(da: DataArray, prcp: DataArray, coord: str = 'date') -> float:

    # rename precip coordinate name (to avoid problems with 'index' or 'date')
    prcp = prcp.rename({list(prcp.coords.keys())[0]: coord})

    # slice prcp to the same time window as the discharge
    prcp = prcp.sel({coord: slice(da.coords[coord][0], da.coords[coord][-1])})

    # determine the date of the first October 1st in the data period
    first_date = da.coords[coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date > datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d'):
        start_date = datetime.strptime(f'{first_date.year + 1}-10-01', '%Y-%m-%d')
    else:
        start_date = datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d')

    end_date = start_date + relativedelta(years=1) - relativedelta(days=1)

    # mask only valid time steps (only discharge has missing values)
    idx = (da >= 0) & (~da.isnull())
    da = da[idx]
    prcp = prcp[idx]

    # calculate long-term means
    q_mean_total = da.mean()
    p_mean_total = prcp.mean()

    values = []
    while end_date < last_date:
        q = da.sel({coord: slice(start_date, end_date)})
        p = prcp.sel({coord: slice(start_date, end_date)})

        val = (q.mean() - q_mean_total) / (p.mean() - p_mean_total) * (p_mean_total / q_mean_total)
        values.append(val)

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.median([float(v) for v in values])
