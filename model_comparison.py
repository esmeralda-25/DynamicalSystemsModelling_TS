import numpy as np
import pandas as pd
from parameter_fitting import param_fit_grid_search_parallel


def BIC(LL, k, n):
    """
    Compute the Bayesian Information Criterion
    args:
    LL: float, log-likelihood of the model
    k: int, number of free parameters in the model
    n: int, number of observations
    returns:
    BIC: float, Bayesian Information
    """
    return -2 * LL + k * np.log(n)


def model_fit(
    data,
    T,
    x_0,
    g_values,
    c_values,
    alpha_values,
    gamma_values,
    sigma,
    tau_P=1,
    model_names=["base", "perseverance", "feedback", "PF"],
):
    """
    Fit models in model_names to the data provided.
    The function fits the models using grid search and returns the best parameters for each model.
    args:
    data: pd.DataFrame, the data to fit
    T: model simulation time
    x_0: float, initial value of the state variable
    g_values: np.ndarray, values of the g parameter to search over
    c_values: np.ndarray, values of the c parameter to search over
    alpha_values: np.ndarray, values of the alpha parameter to search over
    gamma_values: np.ndarray, values of the gamma parameter to search over
    sigma: float, standard deviation of the noise
    tau_P: float, time constant of the persistence process
    model_names: list of strings, names of the models to fit
    returns:
    results: dict, with the following structure:
        {
            'model_name': {'g': float, 'c': float, 'alpha': float, 'gamma': float, 'LL': float, 'BIC': float},
            ...
        }
    """

    n = len(data)
    results = {}
    if "base" in model_names:
        # fit the base model (alpha and gamma fixed to 0)
        best_params_base, best_LL_base, _ = param_fit_grid_search_parallel(
            data, T, x_0, g_values, c_values, np.zeros(1), np.zeros(1), sigma, tau_P
        )
        best_g_base, best_c_base, best_alpha_base, best_gamma_base = best_params_base
        k_base = 2  # c and g are free parameters
        BIC_base = BIC(best_LL_base, k_base, n)
        results["base"] = {
            "g": best_g_base,
            "c": best_c_base,
            "alpha": best_alpha_base,
            "gamma": best_gamma_base,
            "LL": best_LL_base,
            "BIC": BIC_base,
        }

    if "perseverance" in model_names:
        # fit the persistence model (alpha fixed to 0)
        best_params_P, best_LL_P, _ = param_fit_grid_search_parallel(
            data, T, x_0, g_values, c_values, np.zeros(1), gamma_values, sigma, tau_P
        )
        best_g_P, best_c_P, best_alpha_P, best_gamma_P = best_params_P
        k_P = 3  # c, g, gamma are free parameters
        BIC_P = BIC(best_LL_P, k_P, n)
        results["perseverance"] = {
            "g": best_g_P,
            "c": best_c_P,
            "alpha": best_alpha_P,
            "gamma": best_gamma_P,
            "LL": best_LL_P,
            "BIC": BIC_P,
        }

    if "feedback" in model_names:
        # fit the full model with feedback (gamma fixed to 0)
        best_params_F, best_LL_F, _ = param_fit_grid_search_parallel(
            data, T, x_0, g_values, c_values, alpha_values, np.zeros(1), sigma, tau_P
        )
        best_g_F, best_c_F, best_alpha_F, best_gamma_F = best_params_F
        k_F = 3  # c, g, alpha are free parameters
        BIC_F = BIC(best_LL_F, k_F, n)
        results["feedback"] = {
            "g": best_g_F,
            "c": best_c_F,
            "alpha": best_alpha_F,
            "gamma": best_gamma_F,
            "LL": best_LL_F,
            "BIC": BIC_F,
        }

    if "PF" in model_names:
        # fit the full model with perseverance and feedback (all parameters free to vary)
        best_params_PF, best_LL_PF, _ = param_fit_grid_search_parallel(
            data, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma, tau_P
        )
        best_g_PF, best_c_PF, best_alpha_PF, best_gamma_PF = best_params_PF
        k_PF = 4  # c, g, alpha, gamma are free parameters
        BIC_PF = BIC(best_LL_PF, k_PF, n)
        results["PF"] = {
            "g": best_g_PF,
            "c": best_c_PF,
            "alpha": best_alpha_PF,
            "gamma": best_gamma_PF,
            "LL": best_LL_PF,
            "BIC": BIC_PF,
        }

    return results


def model_comparision_results_statistics(results_df):
    """
    Compute the mean, standard deviation, standard error and 95% confidence interval for each parameter in the results, grouped by the best fitted model.
    args:
    results_df: pd.DataFrame, the results DataFrame
    returns:
    stats: pd.DataFrame, the statistics DataFrame with the following structure:
    """
    results = results_df.copy()

    # group by the best fitted model
    grouped = results.groupby("best_model")

    # compute the mean and standard deviation for each column in each group.
    # the result is a DataFrame with a MultiIndex in columns: (parameter, statistic)
    stats = grouped.agg(["mean", "std"])

    # number of participants per group
    counts = grouped.size()

    # for each parameter, compute se and 95% confidence interval.
    # the se = std / sqrt(n)
    # 95% CI = mean ± 1.96 * se
    for param in stats.columns.levels[0]:
        # Retrieve standard deviation series for the current parameter
        std_series = stats[(param, "std")]
        se = std_series / np.sqrt(counts)
        stats[(param, "se")] = se
        stats[(param, "ci_lower")] = stats[(param, "mean")] - 1.96 * se
        stats[(param, "ci_upper")] = stats[(param, "mean")] + 1.96 * se

    # sort columns by parameter names for clarity
    stats = stats.sort_index(axis=1, level=0)
    # round numeric values
    stats = stats.map(lambda x: np.round(x, 3) if pd.notnull(x) and isinstance(x, (int, float)) else x)
    # add counts to the stats
    stats["count"] = counts

    return stats
