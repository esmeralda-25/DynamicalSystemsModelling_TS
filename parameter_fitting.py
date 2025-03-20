# defining the functions for parameter fitting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import itertools
from tqdm.notebook import tqdm

from model import simulate_dynamics
from model_simulation import compute_choice_probabilities


# defining a function for computing the log likelihoods of the parameters given the data
def compute_log_likelihood(df, T, x_0, g, c, alpha, gamma, sigma, tau_P=1):
    """
    Compute the log likelihood of the parameters given the data.
    args:
    df: pandas DataFrame
    T: float, time interval to simulate the model
    x_0: numpy array, initial conditions for the model
    g: float, gain parameter
    c: float, choice parameter
    alpha: float, learning rate
    gamma: float, perseverance parameter
    sigma: float, noise parameter
    tau_P: float, time constant for the persistence variable P
    returns:
    LL: float, log likelihood of the parameters given the data
    """
    # initialize log likelihood
    x_0 = np.array(x_0, dtype=np.float64, copy=True)
    LL = 0
    choices = df["choice_adjusted"].values
    tasks = df["task"].values
    responses = df["response"].values
    num_trials = len(df)
    # get the integrals values at the beginning and end of the trial, we only need the end values
    num_sample_points_per_trial = 2

    for trial_number in range(num_trials):
        if trial_number == 0:
            feedback = 0
        else:
            feedback = 1 if responses[trial_number] == 1 else -1
        # when computing the likelihood, we don't need to make a choice
        # instead, we observe the choice from the data
        choice = choices[trial_number]

        # get input from df for current trial
        if tasks[trial_number] == "letter":
            input = np.array([1, 0])
        else:
            input = np.array([0, 1])

        _, x1_values, x2_values, P_values = simulate_dynamics(
            T, x_0, g, alpha, gamma, input, feedback, sigma, tau_P, num_sample_points=num_sample_points_per_trial
        )  # run model i.e. solve OED

        # update x_0
        x_0 = np.array([x1_values[-1], x2_values[-1], P_values[-1]], dtype=np.float64, copy=True)

        activities_n_final = x_0[:2]  # take x1 and x2 final activities

        choice_probabilities = compute_choice_probabilities(activities_n_final, trial_number, c)

        LL = LL + np.log(choice_probabilities[choice])

    return LL


# defining a function for grid search of the log likelihods of the paramerers given the data
def param_fit_grid_search(
    df, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma=0.2, tau_P=1, pbar_start_pos=0
):
    """
    Perform a grid search to find the best parameters for the model.
    args:
    df: pandas DataFrame
    T: float, time interval to simulate the model
    x_0: numpy array, initial conditions for the model
    g_values: 1D numpy array, values of the gain parameter to search
    c_values: 1D numpy array, values of the choice parameter to search
    alpha_values: 1D numpy array, values of the learning rate to search
    gamma_values: 1D numpy array, values of the perseverance parameter to search
    sigma: float, noise parameter
    tau_P: float, time constant for the persistence variable P
    pbar_start_pos: int, starting position for the progress bar
    returns:
    tuple: tuple of parameter with maximum LL, (best_g, best_c, best_alpha, best_gamma)
    best_LL: float, log likelihood of the best parameters
    LL_matrix: 4D numpy array, log likelihood values for all parameter combinations
    """

    num_c, num_g, num_alpha, num_gamma = len(c_values), len(g_values), len(alpha_values), len(gamma_values)
    LL_matrix = np.zeros(shape=(num_g, num_c, num_alpha, num_gamma))
    best_LL = -np.inf
    best_c = 0
    best_g = 0
    best_alpha = 0
    best_gamma = 0

    # loop through each parameter combination (grid search)
    for g_idx, g in tqdm(enumerate(c_values), total=num_c, desc="c", leave=False, position=pbar_start_pos):
        for c_idx, c in tqdm(enumerate(g_values), total=num_g, desc="g", leave=False, position=pbar_start_pos + 1):
            for alpha_idx, alpha in tqdm(
                enumerate(alpha_values), total=num_alpha, desc="alpha", leave=False, position=pbar_start_pos + 2
            ):
                for gamma_idx, gamma in tqdm(
                    enumerate(gamma_values), total=num_gamma, desc="gamma", leave=False, position=pbar_start_pos + 3
                ):
                    # compute log likelihood
                    LL = compute_log_likelihood(df, T, x_0, g, c, alpha, gamma, sigma, tau_P)

                    # store log likelihood in the data matrix
                    LL_matrix[g_idx, c_idx, alpha_idx, gamma_idx] = LL

                    if LL > best_LL:
                        best_g = g
                        best_c = c
                        best_alpha = alpha
                        best_gamma = gamma
                        best_LL = LL

    return (best_g, best_c, best_alpha, best_gamma), best_LL, LL_matrix


def plot_heatmap(matrix, x_values, y_values, x_label, y_label, title="", ax=None):
    """
    Plot a heatmap of the given matrix.
    args:
    matrix: 2D numpy array, matrix to plot
    x_values: 1D numpy array, values for the x-axis
    y_values: 1D numpy array, values for the y-axis
    x_label: str, label for the x-axis
    y_label: str, label for the y-axis
    title: (optional) str, title of the plot
    ax: (optional) matplotlib axis, axis to plot on
    returns:
    ax: matplotlib axis
    """
    show_plot = False
    if ax is None:
        ax = plt.gca()
        show_plot = True

    img = ax.imshow(
        matrix,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
    )
    plt.colorbar(img, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if show_plot:
        plt.draw()

    return ax


# region parallelized parameter fitting

import numpy as np
import itertools
from tqdm.notebook import tqdm
from multiprocessing import cpu_count
from functools import partial


def param_fit_grid_search_parallel(
    df, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma=0.2, tau_P=1, n_jobs=-1
):
    """
    Perform a grid search to find the best parameters for the model in parallel.
    args:
    df: pandas DataFrame
    T: float, time interval to simulate the model
    x_0: numpy array, initial conditions for the model
    g_values: 1D numpy array, values of the gain parameter to search
    c_values: 1D numpy array, values of the choice parameter to search
    alpha_values: 1D numpy array, values of the learning rate to search
    gamma_values: 1D numpy array, values of the perseverance parameter to search
    sigma: float, noise parameter
    tau_P: float, time constant for the persistence variable P
    n_jobs: int, number of parallel jobs to run
    returns:
    tuple: tuple of parameter with maximum LL, (best_g, best_c, best_alpha, best_gamma)
    best_LL: float, log likelihood of the best parameters
    LL_matrix: 4D numpy array, log likelihood values for all parameter combinations
    """

    # Ensure parameters are numpy arrays
    g_values = np.array(g_values)
    c_values = np.array(c_values)
    alpha_values = np.array(alpha_values)
    gamma_values = np.array(gamma_values)

    param_sets = (g_values, c_values, alpha_values, gamma_values)

    # Total number of combinations
    total_combinations = len(g_values) * len(c_values) * len(alpha_values) * len(gamma_values)

    # Create a partial function with the shared arguments bound.
    x_0 = np.array(x_0, dtype=np.float64, copy=True)
    log_likelihood_partial = partial(compute_log_likelihood, df, T, x_0.copy(), sigma=sigma, tau_P=tau_P)

    def process_chunk(start_idx, end_idx, param_sets):
        """
        Worker function to process a range of parameter combinations lazily.
        """
        g_values, c_values, alpha_values, gamma_values = param_sets
        results = []

        # Generate only the necessary combinations within this range
        total_combinations = itertools.product(g_values, c_values, alpha_values, gamma_values)

        for idx, (g, c, alpha, gamma) in enumerate(total_combinations):
            if start_idx <= idx < end_idx:
                LL = log_likelihood_partial(g, c, alpha, gamma)
                results.append((LL, g, c, alpha, gamma))

        return results

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = min(cpu_count(), total_combinations)  # Don't use more cores than needed

    # Compute range for each worker
    chunk_size = total_combinations // n_jobs
    ranges = [(i * chunk_size, (i + 1) * chunk_size if i < n_jobs - 1 else total_combinations) for i in range(n_jobs)]

    # Parallel execution: each worker processes a *disjoint range* of combinations
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(start, end, param_sets)
        for start, end in tqdm(ranges, desc="Parallel Grid Search", leave=False)
    )

    # Flatten results
    results = [item for sublist in results for item in sublist]

    # Initialize matrix
    LL_matrix = np.zeros((len(g_values), len(c_values), len(alpha_values), len(gamma_values)))
    best_LL = -np.inf
    best_params = None  # (g, c, alpha, gamma)

    # Populate matrix
    for LL, g, c, alpha, gamma in results:
        g_idx = np.where(g_values == g)[0][0]
        c_idx = np.where(c_values == c)[0][0]
        alpha_idx = np.where(alpha_values == alpha)[0][0]
        gamma_idx = np.where(gamma_values == gamma)[0][0]

        LL_matrix[g_idx, c_idx, alpha_idx, gamma_idx] = LL

        if LL > best_LL:
            best_LL = LL
            best_params = (g, c, alpha, gamma)

    return best_params, best_LL, LL_matrix
