import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

import dask.dataframe as dd

def run_gp(ID, plot=True):
    df_id = df_alerts_pandas[df_alerts_pandas['objectId'] == ID]
    minjd = df_id['jd'].min()
    t_band0 = (df_id[df_id['fid'] == 1]['jd']-minjd).to_numpy()
    t_band1 = (df_id[df_id['fid'] == 2]['jd']-minjd).to_numpy()
    mag_band0 = (df_id[df_id['fid'] == 1]['magpsf']).to_numpy()
    mag_band1 = (df_id[df_id['fid'] == 2]['magpsf']).to_numpy()
    magerr_band0 = np.zeros_like(mag_band0)
    magerr_band1 = np.zeros_like(mag_band1)

    # Prediction grid
    min_t, max_t = min(t_band0.min(), t_band1.min()), max(t_band0.max(), t_band1.max())
    # t_pred = np.arange(min_t - 0.5, max_t + 0.5, 2)
    t_pred = np.sort(np.concatenate((t_band0, t_band1)))

    # Independent GP per band
    mu_pred, std_pred, exec_time = [], [], []

    kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

    for t_b, y_b, yerr_b in [(t_band0, mag_band0, magerr_band0),
                             (t_band1, mag_band1, magerr_band1)]:

        X = t_b.reshape(-1,1)
    #     kernel = 1.0*Matern(length_scale=1.5, nu=2.5) + WhiteKernel(noise_level=0.01)
        
        # Suppressing the convergence warning as per https://news.ycombinator.com/item?id=26294156
        _times = []
        for _ in range(5):
            start_time = time.time()
            gp = GaussianProcessRegressor(kernel=kernel, alpha=yerr_b**2, normalize_y=True)
            gp.fit(X, y_b)
            end_time = time.time()
            mu, std = gp.predict(t_pred.reshape(-1,1), return_std=True)
            _times.append(end_time - start_time)
        median_exec_time = np.median(_times)

        exec_time.append(median_exec_time)
        mu_pred.append(mu)
        std_pred.append(std)

    if plot:
        fig = plt.figure(figsize=(10,4))
        ax = plt.gca()
        for i, (t_b, y_b) in enumerate([(t_band0, mag_band0),(t_band1, mag_band1)]):
            offset = -2.5*i
            ax.errorbar(t_b, y_b+offset, yerr=(magerr_band0 if i==0 else magerr_band1),
                         fmt='.', ms=5, label=f'band {i+1} obs')
            ax.plot(t_pred, mu_pred[i]+offset, label=f'band {i+1} mean')
            ax.fill_between(t_pred, mu_pred[i]-2*std_pred[i]+offset, mu_pred[i]+2*std_pred[i]+offset, alpha=0.25)

        ax.invert_yaxis()
        ax.set_xlabel("time (days)")
        ax.set_ylabel("magnitude (offset per band)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    return mu_pred, std_pred, np.sum(exec_time)

from functools import partial
from typing import Dict, List, Union

import george
import numpy as np
import pandas as pd
import scipy.optimize as op
from astropy.table import Table, vstack

pb_wavelengths = {
    1: 4802.0,
    2: 6231.0,
}
def fit_2d_gp(
    obj_data: pd.DataFrame,
    return_kernel: bool = False,
    pb_wavelengths: Dict = pb_wavelengths,
    **kwargs,
):
    """Fit a 2D Gaussian process.

    If required, predict the GP at evenly spaced points along a light curve.

    Parameters
    ----------
    obj_data : pd.DataFrame
        Time, flux and flux error of the data (specific filter of an object).
    return_kernel : bool, default = False
        Whether to return the used kernel.
    pb_wavelengths: dict
        Mapping of the passband wavelengths for each filter used.
    kwargs : dict
        Additional keyword arguments that are ignored at the moment. We allow
        additional keyword arguments so that the various functions that
        call this one can be called with the same arguments.

    Returns
    -------
    kernel: george.gp.GP.kernel, optional
        The kernel used to fit the GP.
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.

    Examples
    --------
    >>> gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    >>> inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}
    >>> gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
    ...
    """
    guess_length_scale = 20.0  # a parameter of the Matern32Kernel

    obj_times = obj_data.jd.astype(float)
    obj_flux = obj_data.magpsf.astype(float)
    obj_flux_error = obj_data.sigmapsf.astype(float)
    obj_wavelengths = obj_data["fid"].map(pb_wavelengths)

    def neg_log_like(p):  # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p):  # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    # Use the highest signal-to-noise observation to estimate the scale. We
    # include an error floor so that in the case of very high
    # signal-to-noise observations we pick the maximum flux value.
    signal_to_noises = np.abs(obj_flux) / np.sqrt(
        obj_flux_error**2 + (1e-2 * np.max(obj_flux)) ** 2
    )
    scale = np.abs(obj_flux[signal_to_noises.idxmax()])

    kernel = (0.5 * scale) ** 2 * george.kernels.Matern32Kernel(
        [guess_length_scale**2, 6000**2], ndim=2
    )
    kernel.freeze_parameter("k2:metric:log_M_1_1")

    gp = george.GP(kernel)
    default_gp_param = gp.get_parameter_vector()
    x_data = np.vstack([obj_times, obj_wavelengths]).T
    gp.compute(x_data, obj_flux_error)

    bounds = [(0, np.log(1000**2))]
    bounds = [(default_gp_param[0] - 10, default_gp_param[0] + 10)] + bounds
    results = op.minimize(
        neg_log_like,
        gp.get_parameter_vector(),
        jac=grad_neg_log_like,
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-6,
    )

    if results.success:
        gp.set_parameter_vector(results.x)
    else:
        # Fit failed. Print out a warning, and use the initial guesses for fit
        # parameters.
        obj = obj_data["object_id"][0]
        print("GP fit failed for {}! Using guessed GP parameters.".format(obj))
        gp.set_parameter_vector(default_gp_param)

    gp_predict = partial(gp.predict, obj_flux)

    if return_kernel:
        return kernel, gp_predict
    return gp_predict


def predict_2d_gp(gp_predict, gp_times, gp_wavelengths):
    """Outputs the predictions of a Gaussian Process.

    Parameters
    ----------
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    gp_times : numpy.ndarray
        Times to evaluate the Gaussian Process at.
    gp_wavelengths : numpy.ndarray
        Wavelengths to evaluate the Gaussian Process at.

    Returns
    -------
    obj_gps : pandas.core.frame.DataFrame, optional
        Time, flux and flux error of the fitted Gaussian Process.

    Examples
    --------
    >>> gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
    >>> number_gp = timesteps
    >>> gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), number_gp)
    >>> obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    >>> obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)
    ...
    """
    unique_wavelengths = np.unique(gp_wavelengths)
    number_gp = len(gp_times)
    obj_gps = []
    for wavelength in unique_wavelengths:
        gp_wavelengths = np.ones(number_gp) * wavelength
        pred_x_data = np.vstack([gp_times, gp_wavelengths]).T
        pb_pred, pb_pred_var = gp_predict(pred_x_data, return_var=True)
        # stack the GP results in a array momentarily
        obj_gp_pb_array = np.column_stack((gp_times, pb_pred, np.sqrt(pb_pred_var)))
        obj_gp_pb = Table(
            [
                obj_gp_pb_array[:, 0],
                obj_gp_pb_array[:, 1],
                obj_gp_pb_array[:, 2],
                [wavelength] * number_gp,
            ],
            names=["jd", "magpsf", "sigmapsf", "fid"],
        )
        if len(obj_gps) == 0:  # initialize the table for 1st passband
            obj_gps = obj_gp_pb
        else:  # add more entries to the table
            obj_gps = vstack((obj_gps, obj_gp_pb))

    obj_gps = obj_gps.to_pandas()
    return obj_gps

if __name__ == '__main__':
    test_objIds = np.load('evaluate_test_objIds_dataloader.npy')  # it's order matches test_outputs_condensed
    #IDs = ['ZTF19aagwxlv', 'ZTF18abtrubp']
    df_alerts = dd.read_parquet(
        'alerts_processed_ftransfer_ztf_2025_05_31_430518.parquet',
        columns=[
                'objectId', 'finkclass', 'tnsclass',
                 'fid', 'magpsf', 'sigmapsf', 'jd'
            ]
    )
    df_alerts_pandas = df_alerts.compute()
    total_time = []
    for ID in test_objIds:
        df = df_alerts_pandas[df_alerts_pandas['objectId'] == ID]

        start = time.time()
        gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
        end = time.time()

        gp_times = np.arange(min(df["jd"]), max(df["jd"]), 2)  # 2 days
        filters = df["fid"]
        filters = list(np.unique(filters))
        gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
        obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)

        #end = time.time()

        inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}
        obj_gps["fid"] = obj_gps["fid"].map(inverse_pb_wavelengths)
        #for _ in range(5):
        
        #lc_times.append(t)
        #exec_time = np.median(lc_times)
        num_obs = len(df)
        total_time.append([ID, end - start, num_obs])
    np.save('gp_exec_time_vs_nobs.npy', total_time)
