import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from ctypes import ArgumentError
from scipy.stats import pearsonr

from model_simulation import simulate_experiment
from parameter_fitting import param_fit_grid_search_parallel


def param_recovery(
    num_trials,
    T,
    x_0,
    to_recover_g,
    to_recover_c,
    to_recover_alpha,
    to_recover_gamma,
    g_values,
    c_values,
    alpha_values,
    gamma_values,
    sigma=0.2,
    tau_P=1,
):
    """
    Recover parameters from simulated data.
    args:
    num_trials: int, number of trials in the experiment
    T: float, time interval to simulate the model
    x_0: numpy array, initial conditions for the model
    to_recover_g: np.ndarray, values of g to recover
    to_recover_c: np.ndarray, values of c to recover
    to_recover_alpha: np.ndarray, values of alpha to recover
    to_recover_gamma: np.ndarray, values of gamma to recover
    g_values: np.ndarray, values of g to fit
    c_values: np.ndarray, values of c to fit
    alpha_values: np.ndarray, values of alpha to fit
    gamma_values: np.ndarray, values of gamma to fit
    sigma: float, standard deviation of the noise
    tau_P: float, time constant for the persistence variable P
    returns:
    fitted_g_log: np.ndarray, fitted values of g
    fitted_c_log: np.ndarray, fitted values of c
    fitted_alpha_log: np.ndarray, fitted values of alpha
    fitted_gamma_log: np.ndarray, fitted values of gamma
    LL_log: np.ndarray, log likelihood
    """
    # make sure all parameters to recover have the same number of values
    num_true_param_set = len(to_recover_g)
    if (
        len(to_recover_c) != num_true_param_set
        or len(to_recover_alpha) != num_true_param_set
        or len(to_recover_gamma) != num_true_param_set
    ):
        raise ArgumentError(
            "you have to give the same amount of values for all parameters to recover (g, c, alpha, gamma)"
        )

    # Create arrays for logging true and fitted parameters
    true_g_log = np.zeros(num_true_param_set)
    true_c_log = np.zeros(num_true_param_set)
    true_alpha_log = np.zeros(num_true_param_set)
    true_gamma_log = np.zeros(num_true_param_set)

    fitted_g_log = np.zeros(num_true_param_set)
    fitted_c_log = np.zeros(num_true_param_set)
    fitted_alpha_log = np.zeros(num_true_param_set)
    fitted_gamma_log = np.zeros(num_true_param_set)
    LL_log = np.zeros(num_true_param_set)

    # For each run, simulate data with randomly sampled parameters and attempt to fit those data
    for run in tqdm(range(num_true_param_set), total=num_true_param_set, desc="run", leave=True, position=0):

        # Randomly sample model parameters from which to sample surrogate data
        true_g = to_recover_g[run]
        true_c = to_recover_c[run]
        true_alpha = to_recover_alpha[run]
        true_gamma = to_recover_gamma[run]

        # Log those parameters
        true_g_log[run] = true_g
        true_c_log[run] = true_c
        true_alpha_log[run] = true_alpha
        true_gamma_log[run] = true_gamma

        # Simulate surrogate data
        df = simulate_experiment(num_trials, T, x_0, true_g, true_c, true_alpha, true_gamma, sigma, tau_P)
        best_params, best_LL, LL_matrix = param_fit_grid_search_parallel(
            df, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma, tau_P
        )

        best_g, best_c, best_alpha, best_gamma = best_params
        # Store the parameter combination yielding the highest log likelihood
        fitted_g_log[run] = best_g
        fitted_c_log[run] = best_c
        fitted_alpha_log[run] = best_alpha
        fitted_gamma_log[run] = best_gamma
        LL_log[run] = best_LL

    return fitted_g_log, fitted_c_log, fitted_alpha_log, fitted_gamma_log, LL_log


def summarize_paramRecovery_results(
    param_names, true_g, true_c, true_alpha, true_gamma, fitted_g, fitted_c, fitted_alpha, fitted_gamma, LL
):
    """
    Summarize the results of the parameter recovery.
    args:
    param_names: list of strings, names of the parameters
    true_g: np.ndarray, true values of g
    true_c: np.ndarray, true values of c
    true_alpha: np.ndarray, true values of alpha
    true_gamma: np.ndarray, true values of gamma
    fitted_g: np.ndarray, fitted values of g
    fitted_c: np.ndarray, fitted values of c
    fitted_alpha: np.ndarray, fitted values of alpha
    fitted_gamma: np.ndarray, fitted values of gamma
    LL: np.ndarray, log likelihood
    returns:
    results_df: pandas DataFrame, results of the parameter recovery
    """
    # make a df of the results
    results_df = pd.DataFrame(
        {
            "true_g": true_g,
            "true_c": true_c,
            "true_alpha": true_alpha,
            "true_gamma": true_gamma,
            "fitted_g": fitted_g,
            "fitted_c": fitted_c,
            "fitted_alpha": fitted_alpha,
            "fitted_gamma": fitted_gamma,
            "LL": LL,
        }
    )

    # calculate the Pearson correlation coefficient for each parameter
    corr_true_fitted = [pearsonr(results_df[f"true_{param}"], results_df[f"fitted_{param}"]) for param in param_names]
    # results of base model
    inter_corr = pd.DataFrame(corr_true_fitted, columns=["correlation", "p-value"], index=param_names, dtype=float)
    inter_corr.index.name = "parameter"
    # round correlation values
    inter_corr["correlation"] = inter_corr["correlation"].round(3)
    # convert p values into scientific notation
    inter_corr["p-value"] = inter_corr["p-value"].apply(lambda x: "%.2e" % x)
    # calculate pairwise cross correlation between the parameters (fitted values of different parameters)
    cross_corr = results_df[["fitted_g", "fitted_c", "fitted_alpha", "fitted_gamma"]].corr(
        method=lambda x, y: pearsonr(x, y)[0]
    )
    return results_df, inter_corr, cross_corr


# region plotting


def plot_true_fitted_correlation(true_params, fitted_params, param_range, margin=0.1, ax=None):
    """
    Plot the correlation between true and fitted parameters.
    args:
    true_params: np.ndarray, true parameter values
    fitted_params: np.ndarray, fitted parameter values
    param_range: tuple, range of the parameters
    margin: float, margin to add to the plot
    ax: (optional) axis to plot on
    returns:
    ax: axis
    """
    show_plot = False
    if ax is None:
        ax = plt.gca()
        show_plot = True
    ax.scatter(true_params, fitted_params, marker="o", color="red")
    ax.set_xlabel("True")
    ax.set_ylabel("Fitted")
    pad = margin * (np.max(param_range) - np.min(param_range))
    ax.set_xlim([np.min(param_range) - pad, np.max(param_range) + pad])
    ax.set_ylim([np.min(param_range) - pad, np.max(param_range) + pad])
    # ax.set_aspect('equal')
    if show_plot:
        plt.draw()
    return ax


def plot_cross_correlation(cross_corr, param_names, ax=None):
    """
    Plot the cross correlation between the fitted parameters.
    args:
    cross_corr: pd.DataFrame, cross correlation between the fitted parameters
    param_names: list of strings, names of the parameters
    ax: (optional) axis to plot on
    returns:
    ax: axis
    """
    show_plot = False
    if ax is None:
        ax = plt.gca()
        show_plot = True
    im = ax.imshow(cross_corr, cmap="coolwarm", vmin=-1, vmax=1)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            ax.text(j, i, cross_corr.iloc[i, j].round(2), ha="center", va="center", color="black")
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45)
    ax.set_yticklabels(param_names)
    plt.colorbar(im, ax=ax, shrink=0.75)

    if show_plot:
        plt.draw()
    return ax
