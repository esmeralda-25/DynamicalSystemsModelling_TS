import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ctypes import ArgumentError
from tqdm.notebook import tqdm

from model_simulation import simulate_experiment
from parameter_fitting import param_fit_grid_search_parallel
from model_comparison import model_fit, BIC


def model_recovery(
    num_trials,
    T,
    x_0,
    sampled_g,
    sampled_c,
    sampled_alpha,
    sampled_gamma,
    g_values,
    c_values,
    alpha_values,
    gamma_values,
    sigma=0.2,
    tau_P=1,
    model_names=["base", "perseverance", "feedback", "PF"],
):
    """ "
    Function to recover the models from simulated data.
    The function will simulate data with the sampled parameters and fit the models to the data.
    args:
    num_trials: int, number of trials in the experiment
    T: int, model simulation time
    x_0: float, initial value of the state variable
    sampled_g: np.ndarray, sampled values of the g parameter
    sampled_c: np.ndarray, sampled values of the c parameter
    sampled_alpha: np.ndarray, sampled values of the alpha parameter
    sampled_gamma: np.ndarray, sampled values of the gamma parameter
    g_values: np.ndarray, values of the g parameter to search over
    c_values: np.ndarray, values of the c parameter to search over
    alpha_values: np.ndarray, values of the alpha parameter to search over
    gamma_values: np.ndarray, values of the gamma parameter to search over
    sigma: float, standard deviation of the noise
    tau_P: float, time constant of the persistence process
    model_names: list of strings, names of the models to fit
    returns:
    confusion_matrix: np.ndarray, confusion matrix of the model recovery
    inverse_matrix: np.ndarray, inversion matrix of the model recovery
    true_fitted_df: pd.DataFrame, DataFrame with the true and fitted models
    """

    num_true_param_set = len(sampled_g)
    num_models = len(model_names)
    if (
        len(sampled_c) != num_true_param_set
        or len(sampled_alpha) != num_true_param_set
        or len(sampled_gamma) != num_true_param_set
    ):
        raise ArgumentError(
            "you have to give the same amount of values for all parameters to recover (g, c, alpha, gamma)"
        )

    # section the sampled parameters into num_models splits, each split will be used for one model
    len_equal_shares = num_true_param_set - num_true_param_set % num_models
    num_runs = len_equal_shares // num_models

    models_g = np.array_split(sampled_g[:len_equal_shares], num_models)
    models_c = np.array_split(sampled_c[:len_equal_shares], num_models)
    modes_alpha = np.array_split(sampled_alpha[:len_equal_shares], num_models)
    models_gamma = np.array_split(sampled_gamma[:len_equal_shares], num_models)

    # array for logging true and fitted models
    true_model = np.zeros(num_runs * num_models, dtype=int)
    fitted_model = np.zeros(num_runs * num_models, dtype=int)

    for run in tqdm(range(num_runs), total=num_runs, desc="run", leave=True, position=0):
        # simulate data with the sampled parameters and attempt to fit those data
        for model in range(num_models):
            true_g = models_g[model][run]
            true_c = models_c[model][run]
            true_alpha = modes_alpha[model][run]
            true_gamma = models_gamma[model][run]

            # simulate data
            df = simulate_experiment(num_trials, T, x_0, true_g, true_c, true_alpha, true_gamma, sigma, tau_P)

            # fit all models to the data
            BIC_values = []
            if "base" in model_names:
                _, LL_base, _ = param_fit_grid_search_parallel(
                    df, T, x_0, g_values, c_values, np.zeros(1), np.zeros(1), sigma, tau_P
                )
                BIC_values.append(BIC(LL_base, 2, num_trials))
            if "perseverance" in model_names:
                _, LL_P, _ = param_fit_grid_search_parallel(
                    df, T, x_0, g_values, c_values, np.zeros(1), gamma_values, sigma, tau_P
                )
                BIC_values.append(BIC(LL_P, 3, num_trials))
            if "feedback" in model_names:
                _, LL_F, _ = param_fit_grid_search_parallel(
                    df, T, x_0, g_values, c_values, alpha_values, np.zeros(1), sigma, tau_P
                )
                BIC_values.append(BIC(LL_F, 3, num_trials))
            if "PF" in model_names:
                _, LL_PF, _ = param_fit_grid_search_parallel(
                    df, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma, tau_P
                )
                BIC_values.append(BIC(LL_PF, 4, num_trials))

            BIC_values = np.array(BIC_values)

            # log the true and fitted models
            true_model[model + num_models * run] = model
            fitted_model[model + num_models * run] = np.argmin(BIC_values)

    best_fit_matrix = np.zeros((num_models, num_models))
    for true, fit in zip(true_model, fitted_model):
        best_fit_matrix[int(true), int(fit)] += 1
    best_fit_matrix = best_fit_matrix.T  # transpose to get true at columns and fitted at rows
    confusion_matrix = best_fit_matrix / np.sum(best_fit_matrix, axis=0)
    inverse_matrix = best_fit_matrix / np.sum(best_fit_matrix, axis=1, keepdims=True)

    # replace the ids of the models in the true and fitted model arrays with the actual model ids passed in models
    true_model = np.array([model_names[model] for model in true_model])
    fitted_model = np.array([model_names[model] for model in fitted_model])
    # df of true and fitted models
    true_fitted_df = pd.DataFrame({"true_model": true_model, "fitted_model": fitted_model})

    return confusion_matrix, inverse_matrix, true_fitted_df


def plot_model_recovery(confusion_matrix, inverse_matrix, model_names, axs=None):
    """
    Function to plot the confusion matrix and inversion matrix of the model recovery.
    args:
    confusion_matrix: np.ndarray, confusion matrix of the model recovery
    inverse_matrix: np.ndarray, inversion matrix of the model recovery
    model_names: list of strings, names of the models
    axs: (optional) list of matplotlib axes, axes to plot the confusion matrix and inversion matrix
    returns:
    axs: list of matplotlib axes, axes with the confusion matrix and inversion matrix
    """
    show_plot = False
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        show_plot = True

    num_models = len(model_names)
    axs[0].imshow(confusion_matrix, cmap="viridis", vmin=0, vmax=1)
    axs[0].set_title("Confusion matrix")
    axs[0].set_xlabel("True model")
    axs[0].set_ylabel("Fitted model")
    axs[0].set_xticks(range(num_models))
    axs[0].set_xticklabels(model_names, rotation=45)
    axs[0].set_yticks(range(num_models))
    axs[0].set_yticklabels(model_names, rotation=45)
    # show the the actual values
    for i in range(num_models):
        for j in range(num_models):
            axs[0].text(j, i, f"{confusion_matrix[i, j]:.2f}", ha="center", va="center", color="black")

    im = axs[1].imshow(inverse_matrix, cmap="viridis", vmin=0, vmax=1)
    axs[1].set_title("Inversion matrix")
    axs[1].set_xlabel("True model")
    # axs[1].set_ylabel("Fitted model")
    axs[1].set_xticks(range(num_models))
    axs[1].set_xticklabels(model_names, rotation=45)
    axs[1].set_yticks(range(num_models))
    axs[1].set_yticklabels(model_names, rotation=45)
    for i in range(num_models):
        for j in range(num_models):
            axs[1].text(j, i, f"{inverse_matrix[i, j]:.2f}", ha="center", va="center", color="black")

    # add colorbar 0-1
    cbar = plt.colorbar(im, ax=axs.tolist(), orientation="vertical", shrink=0.70)
    cbar.set_label("Accuracy")

    if show_plot:
        plt.draw()

    return axs
