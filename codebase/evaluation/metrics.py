"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import numpy as np
from scipy import stats, signal
from xarray.core.dataarray import DataArray


def get_available_metrics() -> list:
    """Get list of available metrics."""
    metrics = [
        "NSE", "MSE", "RMSE", "KGE", "Alpha-NSE", "Pearson r", "Beta-NSE", "FHV", "FMS", "FLV", "Peak-Timing"
    ]
    return metrics


def _validate_inputs(obs: DataArray, sim: DataArray):
    if obs.shape != sim.shape:
        raise RuntimeError("Shapes of observations and simulations must match")

    if (len(obs.shape) > 1) and (obs.shape[1] > 1):
        raise RuntimeError(
            "Metrics only defined for time series (1d or 2d with second dimension 1)")


def _mask_valid(obs: DataArray, sim: DataArray) -> (DataArray, DataArray):
    # mask of invalid entries
    idx = (obs >= 0) & (~obs.isnull())

    obs = obs[idx]
    sim = sim[idx]

    return obs, sim


def _get_fdc(da: DataArray) -> np.ndarray:
    return da.sortby(da, ascending=False).values


def nse(obs: DataArray, sim: DataArray) -> float:

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    denominator = ((obs - obs.mean())**2).sum()
    numerator = ((sim - obs)**2).sum()

    value = 1 - numerator / denominator

    return float(value)


def mse(obs: DataArray, sim: DataArray) -> float:

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    return float(((sim - obs)**2).mean())


def rmse(obs: DataArray, sim: DataArray) -> float:

    return np.sqrt(mse(obs, sim))


def alpha_nse(obs: DataArray, sim: DataArray) -> float:

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    return float(sim.std() / obs.std())


def beta_nse(obs: DataArray, sim: DataArray) -> float:

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    return float((sim.mean() - obs.mean()) / obs.std())


def beta_kge(obs: DataArray, sim: DataArray) -> float:

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    return float(sim.mean() / obs.mean())


def kge(obs: DataArray, sim: DataArray, weights: list = [1, 1, 1]) -> float:
    if len(weights) != 3:
        raise ValueError("Weights of the KGE must be a list of three values")

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    r, _ = stats.pearsonr(obs.values, sim.values)

    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()

    value = (weights[0] * (r - 1)**2 + weights[1] * (alpha - 1)**2 + weights[2] * (beta - 1)**2)

    return 1 - np.sqrt(float(value))


def pearsonr(obs: DataArray, sim: DataArray) -> float:
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    r, _ = stats.pearsonr(obs.values, sim.values)

    return float(r)


def fdc_fms(obs: DataArray, sim: DataArray, lower: float = 0.2, upper: float = 0.7) -> float:
    """Slope of the middle section of the flow duration curve.
    
    Reference:
    Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process‐based diagnostic approach to model evaluation: 
    Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, doi:10.1029/2007WR006716. 
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if any([(x <= 0) or (x >= 1) for x in [upper, lower]]):
        raise ValueError("upper and lower have to be in range ]0,1[")

    if lower >= upper:
        raise ValueError("The lower threshold has to be smaller than the upper.")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    # calculate fms part by part
    qsm_lower = np.log(sim[np.round(lower * len(sim)).astype(int)])
    qsm_upper = np.log(sim[np.round(upper * len(sim)).astype(int)])
    qom_lower = np.log(obs[np.round(lower * len(obs)).astype(int)])
    qom_upper = np.log(obs[np.round(upper * len(obs)).astype(int)])

    fms = ((qsm_lower - qsm_upper) - (qom_lower - qom_upper)) / (qom_lower - qom_upper + 1e-6)

    return fms * 100


def fdc_fhv(obs: DataArray, sim: DataArray, h: float = 0.02) -> float:
    """Peak flow bias derived from the flow duration curve.

    Reference:
    Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process‐based diagnostic approach to model evaluation: 
    Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, doi:10.1029/2007WR006716. 
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if (h <= 0) or (h >= 1):
        raise ValueError(
            "h has to be in range ]0,1[. Consider small values, e.g. 0.02 for 2% peak flows")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # subset data to only top h flow values
    obs = obs[:np.round(h * len(obs)).astype(int)]
    sim = sim[:np.round(h * len(sim)).astype(int)]

    fhv = np.sum(sim - obs) / np.sum(obs)

    return fhv * 100


def fdc_flv(obs: DataArray, sim: DataArray, l: float = 0.3) -> float:
    """Low flow bias derived from the flow duration curve.

    Reference:
    Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process‐based diagnostic approach to model evaluation: 
    Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, doi:10.1029/2007WR006716. 
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if (l <= 0) or (l >= 1):
        raise ValueError(
            "l has to be in range ]0,1[. Consider small values, e.g. 0.3 for 30% low flows")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    obs = obs[-np.round(l * len(obs)).astype(int):]
    sim = sim[-np.round(l * len(sim)).astype(int):]

    # transform values to log scale
    obs = np.log(obs)
    sim = np.log(sim)

    # calculate flv part by part
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100


def mean_peak_timing(obs: DataArray, sim: DataArray, window: int = 3,
                     resolution: str = 'D') -> float:
    """Absolute peak timing error
    
    My own metrics: Uses scipy to find peaks in the observed flows and then computes the absolute time difference 
    between the observed flow and the day in which the peak is found in the simulation time series.  

    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    # get indices of peaks and their corresponding height
    peaks, properties = signal.find_peaks(obs.values, distance=100, height=(None, None))

    # evaluate timing
    timing_errors = []
    for idx, height in zip(peaks, properties['peak_heights']):

        # skip peaks at the start and end of the sequence
        if (idx - window < 0) or (idx + window > len(obs)):
            continue

        # check if the value at idx is a peak (both neighbors must be smaller)
        if (sim[idx] > sim[idx - 1]) and (sim[idx] > sim[idx + 1]):
            peak_sim = sim[idx]
        else:
            # define peak around idx as the max value inside of the window
            values = sim[idx - window:idx + window + 1]
            peak_sim = values[values.argmax()]

        # get xarray object of qobs peak, for getting the date and calculating the datetime offset
        peak_obs = obs[idx]

        # calculate the time difference between the peaks
        delta = peak_obs.coords['date'] - peak_sim.coords['date']

        timing_error = np.abs(
            delta.values.astype(f'timedelta64[{resolution}]') / np.timedelta64(1, resolution))

        timing_errors.append(timing_error)

    return np.sum(timing_errors) / len(peaks)


def calculate_all_metrics(obs: DataArray, sim: DataArray) -> dict:
    """Calculate all metrics with default values."""
    results = {
        "NSE": nse(obs, sim),
        "MSE": mse(obs, sim),
        "RMSE": rmse(obs, sim),
        "KGE": kge(obs, sim),
        "Alpha-NSE": alpha_nse(obs, sim),
        "Beta-NSE": beta_nse(obs, sim),
        "Pearson r": pearsonr(obs, sim),
        "FHV": fdc_fhv(obs, sim),
        "FMS": fdc_fms(obs, sim),
        "FLV": fdc_flv(obs, sim),
        "Peak-Timing": mean_peak_timing(obs, sim)
    }

    return results


def calculate_metrics(obs: DataArray, sim: DataArray, metrics: list) -> dict:
    values = {}
    for metric in metrics:
        if metric == 'all':
            values = calculate_all_metrics(obs, sim)
            break
        else:
            if metric.lower() == "nse":
                values["NSE"] = nse(obs, sim)
            elif metric.lower() == "mse":
                values["MSE"] = mse(obs, sim)
            elif metric.lower() == "rmse":
                values["RMSE"] = rmse(obs, sim)
            elif metric.lower() == "kge":
                values["KGE"] = kge(obs, sim)
            elif metric.lower() == "alpha-nse":
                values["Alpha-NSE"] = alpha_nse(obs, sim)
            elif metric.lower() == "beta-nse":
                values["Beta-NSE"] = beta_nse(obs, sim)
            elif metric.lower().replace(" ", "") == "pearsonr":
                values["Pearson r"] = pearsonr(obs, sim)
            elif metric.lower() == "fhv":
                values["FHV"] = fdc_fhv(obs, sim)
            elif metric.lower() == "fms":
                values["FMS"] = fdc_fms(obs, sim)
            elif metric.lower() == "flv":
                values["FLV"] = fdc_flv(obs, sim)
            elif metric.lower() == "peak-timing":
                values["Peak-Timing"] = mean_peak_timing(obs, sim)
            else:
                raise RuntimeError(f"Unknown metric {metric}")

    return values
