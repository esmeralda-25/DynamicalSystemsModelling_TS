import numpy as np
import pandas as pd
from parameter_fitting import param_fit_grid_search_parallel


def BIC(LL, k, n):
    """Compute the Bayesian Information Criterion

    Inputs
    -------
    LL: log likelihood
    k: number of parameters
    n: number of data points
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
    n = len(data)
    print("fitting data to models: ", model_names)
    print("number of data points: ", n)
    results = {}
    if "base" in model_names:
        # Fit the base model (alpha and gamma fixed to 0)
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
        # Fit the persistence model (alpha fixed to 0)
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
        # Fit the full model with feedback (gamma fixed to 0)
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
        # Fit the full model with perseverance and feedback (all parameters free to vary)
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
    Given a DataFrame of participant results (indexed by participant id) with a column 'best_model'
    and other columns corresponding to fit values (LL, BIC, model parameters, etc.),
    this function groups the participants by their best fitted model, and for each parameter computes:
        - mean
        - standard deviation (std)
        - standard error of the mean (se)
        - 95% confidence interval (ci_lower, ci_upper)

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with participant results. It is assumed that there is an index for participant IDs
        and one column named 'best_model' that specifies the winning model.
    Returns
    -------
    stats : pd.DataFrame
        A DataFrame with multi-index columns (parameter, statistic) where each statistic is one of:
        'mean', 'std', 'se', 'ci_lower', 'ci_upper'
    """
    results = results_df.copy()

    # Group by the best fitted model
    grouped = results.groupby("best_model")

    # Compute the mean and standard deviation for each column in each group.
    # The result is a DataFrame with a MultiIndex in columns: (parameter, statistic)
    stats = grouped.agg(["mean", "std"])

    # Number of participants per group
    counts = grouped.size()

    # For each parameter, compute se and 95% confidence interval.
    # The se = std / sqrt(n)
    # 95% CI = mean ± 1.96 * se
    for param in stats.columns.levels[0]:
        # Retrieve standard deviation series for the current parameter
        std_series = stats[(param, "std")]
        se = std_series / np.sqrt(counts)
        stats[(param, "se")] = se
        stats[(param, "ci_lower")] = stats[(param, "mean")] - 1.96 * se
        stats[(param, "ci_upper")] = stats[(param, "mean")] + 1.96 * se

    # Optionally sort columns for clarity (sort by parameter names)
    stats = stats.sort_index(axis=1, level=0)
    # round numeric values
    stats = stats.map(lambda x: np.round(x, 3) if pd.notnull(x) and isinstance(x, (int, float)) else x)
    # add counts to the stats
    stats["count"] = counts

    return stats
