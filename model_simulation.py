import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from model import simulate_dynamics, plot_trajectory


def compute_choice_probabilities(activities_n_final, trial_number, c):
    """
    computes the choice probabilities based on the final activity levels.
    args:
    activities_n_final: np.ndarray, activity of neuron 1 (letter task) and neuron 2 (number task) at the end of trial
    trial_number: int, number of the current trial
    c: float, parameter used to compute beta in the softmax function
    returns:
    choice_probs: np.ndarray, choice probabilities
    """

    # compute beta
    beta = np.power((trial_number + 1) / 10, c)

    # scaling factor (small differences between neuron activities are magnified, which makes the exponential terms more distinct)
    scalar = 3

    # compute softmaxed choice proabbilities
    choice_probs = np.exp(activities_n_final * scalar * beta) / np.sum(np.exp(activities_n_final * scalar * beta))

    return choice_probs


def compute_choice(activities_n_final, trial_number, c):
    """
    computes the choice based on the choice probabilities.
    args:
    activities_n_final: np.ndarray, activity of neuron 1 (letter task) and neuron 2 (number task) at the end of trial
    trial_number: int, number of the current trial
    c: float, parameter used to compute beta in the softmax function
    returns:
    choice: int, chosen deck index
    """
    # compute softmaxed choice proabbilities
    choice_probs = compute_choice_probabilities(activities_n_final, trial_number, c)

    # compute cumulated sums
    cumulated_choice_probs = np.cumsum(choice_probs)

    # draw random number between 0 and 1
    random_number = np.random.random()

    # choose deck index depending on choice probabilities
    choice = 0

    # iterate through the cumulative sums to find the first index where the random number exceeds the cumulative sum
    while choice < len(cumulated_choice_probs) and random_number > cumulated_choice_probs[choice]:
        choice += 1

    return choice


def test_trial_type(current_task, former_task):
    """
    test whether the current task is the same as the former task.
    args:
    current_task: str, current task
    former_task: str, former task
    returns:
    trial_type: str, type of trial (repeat or switch)
    """
    # determining the condition (repeat/ switch)

    if current_task == former_task:
        trial_type = "repeat"
    else:
        trial_type = "switch"

    return trial_type


def initialize_task(task=None):
    """
    initialize the task for the next trial.
    args:
    task: int, task of the current trial
    returns:
    current_task: str, current task
    input: np.ndarray, input to the neural units
    correct_responses: np.ndarray, correct responses for the current task
    """
    # initialization of the task
    if task is None:
        task = random.randint(0, 1)

    if task == 0:
        I1 = 1
        I2 = 0
        input = np.array([I1, I2])
        correct_responses = np.array([1, 0])
        current_task = "letter"

    elif task == 1:
        I1 = 0
        I2 = 1
        input = np.array([I1, I2])
        correct_responses = np.array([0, 1])
        current_task = "number"
    else:
        raise ValueError("Task must be either 0 or 1")

    return current_task, input, correct_responses


def get_correctness(correct_responses, choice):
    """
    get the correctness of the choice.
    args:
    correct_responses: np.ndarray, correct responses for the current task
    choice: int, chosen deck index
    returns:
    correctness: int, correctness of the choice
    """
    # checking if the choice is correct

    if correct_responses[choice] == 1:
        correctness = 1
    else:
        correctness = 0

    return correctness


# generate an experiment with a given number of trials
def generate_experiment_trials(num_trials):
    """
    generate an experiment with a given number of trials.
    args:
    num_trials: int, number of trials
    returns:
    df: pandas DataFrame, data of the experiment
    """
    trials = [initialize_task() for _ in range(num_trials)]
    trails_df = pd.DataFrame(trials, columns=["task", "input", "correct_responses"])

    return trails_df


def generate_experiment_trials_fromData(data):
    """
    generate an experiment with the same sequence of tasks as the provided data.
    args:
    data: pandas DataFrame, data of the experiment / task sequence
    returns:
    df: pandas DataFrame, data of the experiment
    """
    # 0 letter, 1 number
    tasks = data["task"].copy().apply(lambda x: 0 if x == "letter" else 1)
    trials = [initialize_task(task) for task in tasks]
    trails_df = pd.DataFrame(trials, columns=["task", "input", "correct_responses"])

    return trails_df


def log_data(df, trial_number, activities_n_final, current_task, trial_type, choice, correct_responses, correctness):
    """ "
    log the data of the current trial.
    args:
    df, trial_number, activities_n_final, current_task, trial_type, choice, correct_responses, correctness
    returns:
    df: pandas DataFrame, data of the experiment
    """
    # calculate the new index (assuming trial_number starts at 0 and aligns with DataFrame index)
    new_index = len(df)

    # directly assign the new row to the DataFrame using `loc`
    df.loc[new_index] = {
        "trial_index": trial_number,
        "activity_n1": activities_n_final[0],
        "activity_n2": activities_n_final[1],
        "task": current_task,
        "trial_type": trial_type,
        "choice_adjusted": choice,
        "correct_responses": correct_responses,
        "response": correctness,
    }
    return df


def simulate_experiment(
    num_trials,
    T,
    x_0,
    g,
    c,
    alpha,
    gamma,
    sigma,
    tau_P=1,
    num_sample_points_per_trial=100,
    bool_plot_trajectory=False,
    task_sequence=None,
    ax=None,
):
    """
    simulate an experiment with a given number of trials.
    args:
    num_trials: int, number of trials
    T: float, time interval
    x_0: np.ndarray, initial activity of the neurons
    g: float, gain parameter
    c: float, parameter used to compute beta in the softmax function
    alpha: float, decay rate of the neural activity
    gamma: float, decay rate of the working memory
    sigma: float, noise level
    tau_P: float, time constant of the working memory
    num_sample_points_per_trial: int, number of sample points per trial
    bool_plot_trajectory: bool, whether to plot the trajectory
    task_sequence: pandas DataFrame, task sequence
    ax: (optional) matplotlib axis, axis to plot the trajectory
    returns:
    df: pandas DataFrame, data of the experiment
    """

    if task_sequence is not None:
        num_trials = len(task_sequence)
    if task_sequence is None:
        task_sequence = generate_experiment_trials(num_trials)

    # we will log the entire simulation in a dataframe
    df = pd.DataFrame(
        columns=[
            "task",
            "trial_type",
            "response",
            "activity_n1",
            "activity_n2",
            "choice_adjusted",
            "correct_responses",
            "trial_index",
        ]
    )

    # set initial neuron activity
    x_0 = np.asarray(x_0)
    x_1 = x_0[0]
    x_2 = x_0[1]
    P = x_0[2]

    # create arrays for plotting
    array_x1 = []
    array_x2 = []
    array_P = []
    array_ts = []

    feedback = 0  # feedback of first trail is 0 (neutral) correct -> 1, wrong -> -1
    feedback_log = [feedback]
    for trial_number in range(num_trials):

        x_0 = np.array([x_1, x_2, P])

        current_task = task_sequence["task"][trial_number]  # checking wether task is letter or number, input -> [I1,I2]
        correct_responses = task_sequence["correct_responses"][trial_number]  # correct responses for the current task
        input = task_sequence["input"][trial_number]  # input to the neural units

        if trial_number > 0:
            trial_type = test_trial_type(current_task, df["task"].iloc[-1])  # checking wether type is switch or repeat

        else:  # for first trial since there is no former task
            trial_type = None

        ts_values, x1_values, x2_values, P_values = simulate_dynamics(
            T, x_0, g, alpha, gamma, input, feedback, sigma, tau_P, num_sample_points=num_sample_points_per_trial
        )  # run model i.e. solve OED

        activities_n_final = np.array([x1_values[-1], x2_values[-1]])  # get last values of neuron activity

        choice = compute_choice(activities_n_final, trial_number, c)  # computing the choices

        # set the initial x_1 and x_2 for the next trail to be the last values of the current one
        x_1 = x1_values[-1]
        x_2 = x2_values[-1]
        P = P_values[-1]

        correctness = get_correctness(correct_responses, choice)

        # feedback of the current trail
        feedback = 1 if correctness == 1 else -1
        feedback_log.append(feedback)

        # log results
        df = log_data(
            df, trial_number, activities_n_final, current_task, trial_type, choice, correct_responses, correctness
        )

        # plotting part ------------------------------------------------------------

        max_trails_plot = 20
        if bool_plot_trajectory == True:

            if num_trials == 1:
                array_x1 = x1_values
                array_x2 = x2_values
                array_P = P_values
                array_ts = ts_values

            else:
                array_x1 = np.concatenate((array_x1, x1_values), axis=None)
                array_x2 = np.concatenate((array_x2, x2_values), axis=None)
                array_P = np.concatenate((array_P, P_values), axis=None)
                array_ts = np.concatenate((array_ts, ts_values + (T * (trial_number))), axis=None)
    # get rid of the first trial
    df = df.iloc[1:]

    # reset index
    df.reset_index(drop=True, inplace=True)
    df["accuracy"] = (df["response"].cumsum() / (df.index + 1)).astype(float)
    # reorder the dataframe
    df = df[
        [
            "task",
            "trial_type",
            "response",
            "accuracy",
            "choice_adjusted",
            "activity_n1",
            "activity_n2",
            "correct_responses",
            "trial_index",
        ]
    ]

    if (bool_plot_trajectory == True) and (num_trials <= max_trails_plot):
        # get the task sequence in 0 and 1 , input [1,0] -> 0, input [0,1] -> 1
        inputs = task_sequence["input"].apply(lambda x: 0 if x[0] != 1 else 1)
        plot_trajectory(T, array_ts, array_x1, array_x2, array_P, inputs, feedback_log, num_trials, ax=ax)
    elif (bool_plot_trajectory == True) and (num_trials > max_trails_plot):
        print(f"Ploting first {max_trails_plot} trials:")
        inputs = task_sequence["input"].apply(lambda x: 0 if x[0] != 1 else 1)
        plot_trajectory(
            T,
            array_ts[: max_trails_plot * num_sample_points_per_trial],
            array_x1[: max_trails_plot * num_sample_points_per_trial],
            array_x2[: max_trails_plot * num_sample_points_per_trial],
            array_P[: max_trails_plot * num_sample_points_per_trial],
            inputs[:max_trails_plot],
            feedback_log[:max_trails_plot],
            max_trails_plot,
            ax=ax,
        )

    return df
