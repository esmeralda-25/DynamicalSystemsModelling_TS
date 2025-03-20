import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def calculate_accuracy_condition(df):
   """
   Calculates the average accuracy for switch and repeat trials.
   args:
       df: DataFrame containing the data, with columns 'trial_type' and 'response'
   returns:
       dict with keys 'accuracy_switch' and 'accuracy_repeat' containing the average accuracy for switch and repeat trials
   """
   # filter the data for switch and repeat trials
   switch_trials = df[df["trial_type"] == "switch"]
   repeat_trials = df[df["trial_type"] == "repeat"]


   # sum over response values for each condition
   sum_accuracy_switch = switch_trials["response"].sum()
   sum_accuracy_repeat = repeat_trials["response"].sum()


   # get the number of switch and repeat trials
   count_switch = len(switch_trials)
   count_repeat = len(repeat_trials)


   # calculate the average accuracy of each condition
   accuracy_switch = sum_accuracy_switch / count_switch if count_switch > 0 else 0
   accuracy_repeat = sum_accuracy_repeat / count_repeat if count_repeat > 0 else 0


   return {"accuracy_switch": accuracy_switch, "accuracy_repeat": accuracy_repeat}




def plot_accuracy_condition(condition_accuracies, title="", ax=None):
   """
   Plots the average accuracy for switch and repeat trials.
   args:
       condition_accuracies: dict with keys 'accuracy_switch' and 'accuracy_repeat' containing the average accuracy for switch and repeat trials
       title: (optional) title for the plot
       ax: (optional) matplotlib axes to plot on
   returns:
       ax: matplotlib axes containing the plot
   """
   # extract results for plotting
   trial_types = ["Switch", "Repeat"]
   condition_accuracies = [condition_accuracies["accuracy_switch"], condition_accuracies["accuracy_repeat"]]


   # create new axes if none is provided
   show_plot = False
   if ax is None:
       fig, ax = plt.subplots(figsize=(8, 6))
       show_plot = True


   # bar plot on the provided (or new) axis
   bars = ax.bar(trial_types, condition_accuracies, color=["blue", "green"])
   ax.set_xlabel("Trial Type")
   ax.set_ylabel("Accuracy")
   ax.set_title(title)
   ax.set_ylim(0, 1)


   offset = 0.05  # offset from the top of the bar
   for bar in bars:
       height = bar.get_height()
       xpos = bar.get_x() + bar.get_width() / 2.0
       # place the text offset below the top edge of the bar.
       # if the bar is too short, adjust the position so it remains inside.
       text_y = height - offset if height > offset else height * 0.5
       ax.text(xpos, text_y, f"{height:.2f}", ha="center", va="top", color="black")


   # if a new figure was created, display the plot
   if show_plot:
       plt.draw()
   return ax




def calculate_accuracy_across_blocks(data, num_blocks=5):
   """
   Calculates the accuracy for each block of trials.
   args:
       data: DataFrame containing the data, with columns 'trial_type' and 'response'
       num_blocks: number of blocks to divide the data into
   returns:
       DataFrame with columns 'block', 'accuracy_switch', 'accuracy_repeat', 'accuracy'
   """
   data = data.copy()
   # create a 'block' column. Use trial_index if available; otherwise, use the DataFrame's index.
   if "trial_index" in data.columns:
       trial_indices = data["trial_index"]
   else:
       trial_indices = data.index
   # split the trial indices into equal-sized blocks (labels 1...num_blocks)
   data["block"] = pd.cut(trial_indices, bins=num_blocks, labels=range(1, num_blocks + 1))


   # compute block-level accuracy by (mean response within each block)
   overall_block = data.groupby("block", observed=False)["response"].mean().reset_index()
   switch_block = (
       data[data["trial_type"] == "switch"].groupby("block", observed=False)["response"].mean().reset_index()
   )
   repeat_block = (
       data[data["trial_type"] == "repeat"].groupby("block", observed=False)["response"].mean().reset_index()
   )


   # rename response column to accuracy
   overall_block = overall_block.rename(columns={"response": "accuracy"})
   switch_block = switch_block.rename(columns={"response": "accuracy"})
   repeat_block = repeat_block.rename(columns={"response": "accuracy"})


   # return as dataframes of the form
   # block | accuracy_switch | accuracy_repeat | accuracy
   result = pd.merge(overall_block, switch_block, on="block", suffixes=("", "_switch"))
   result = pd.merge(result, repeat_block, on="block", suffixes=("", "_repeat"))
   return result




def plot_accuracy_across_blocks(acc_blocks, num_blocks=5, title="", ax=None):
   """
   Plots the accuracy for each block of trials.
   args:
       acc_blocks: DataFrame with columns 'block', 'accuracy_switch', 'accuracy_repeat', 'accuracy'
       num_blocks: number of blocks to divide the data into
       title: (optional) title for the plot
       ax: (optional) matplotlib axes to plot on
   returns:
       ax: matplotlib axes containing the plot
   """
   show_plot = False
   if ax is None:
       ax = plt.gca()
       show_plot = True


   # plot the results for each condition
   ax.plot(acc_blocks["block"], acc_blocks["accuracy_switch"], marker="o", label="switch")
   ax.plot(acc_blocks["block"], acc_blocks["accuracy_repeat"], marker="o", label="repeat")
   ax.plot(acc_blocks["block"], acc_blocks["accuracy"], marker="o", label="all")


   # Plot chance level
   ax.axhline(0.5, color="black", linestyle="--", label="chance", alpha=0.5, linewidth=0.5)


   # format axes
   ax.set_title(title)
   ax.set_xlabel("Block")
   ax.set_ylabel("Accuracy")
   ax.set_xticks(range(1, num_blocks + 1))
   ax.legend()


   if show_plot:
       plt.draw()
   return ax




def calculate_accuracy_consecutive_condition(df, consecutive_condition, on_condition, max_consecutive=3):
   """
   Calculates the average accuracy for on_condition trials following a streak of consecutive_condition trials.
   args:
       df: DataFrame containing the data, with columns 'trial_type' and 'response'
       consecutive_condition: the trial type whose consecutive occurrences are counted (e.g. "switch" or "repeat")
       on_condition: the trial type for which accuracy is computed when it follows a streak of consecutive_condition trials
       max_consecutive: the maximum number of consecutive trials to consider
   returns:
       DataFrame with columns 'bucket', 'accuracy', 'count'
   """
   df = df.copy()


   # compute a column with the count of consecutive consecutive_condition trials.
   df["consecutive_condition"] = 0
   consecutive_count = 1
   for i in range(len(df)):
       if df.loc[i, "trial_type"] == consecutive_condition:
           df.loc[i, "consecutive_condition"] = consecutive_count
           consecutive_count += 1
       else:
           # When the trial type is not the consecutive_condition, reset the counter.
           df.loc[i, "consecutive_condition"] = 0
           consecutive_count = 1


   # prepare a dictionary to hold responses grouped by preceding streak length (bucket)
   streak_dict = {k: [] for k in range(1, max_consecutive + 1)}


   # iterate over the DataFrame to populate the streak_dict
   for i in range(1, len(df)):
       if df.loc[i, "trial_type"] == on_condition:
           if consecutive_condition == on_condition:
               # when the condition is self-same, the current trial is part of a streak.
               # its current consecutive count is n; it follows a streak of (n - 1).
               # We want to count this trial as following any streak length from 1 to min(n-1, max_consecutive).
               current_streak = df.loc[i, "consecutive_condition"]
               if current_streak > 1:
                   for bucket in range(1, min(current_streak, max_consecutive + 1)):
                       streak_dict[bucket].append(df.loc[i, "response"])
           else:
               # when on_condition differs from consecutive_condition, only count the trial if it
               # immediately follows a consecutive_condition trial.
               if df.loc[i - 1, "trial_type"] == consecutive_condition:
                   # the previous trial’s consecutive count gives the streak length.
                   streak_val = df.loc[i - 1, "consecutive_condition"]
                   bucket = int(min(streak_val, max_consecutive))
                   # only include the trial if bucket is at least 1.
                   if bucket >= 1:
                       streak_dict[bucket].append(df.loc[i, "response"])


   # compute accuracy for each bucket
   buckets = []
   accuracy = []
   counts = []
   for bucket in range(1, max_consecutive + 1):
       buckets.append(bucket)
       if streak_dict[bucket]:
           accuracy.append(np.mean(streak_dict[bucket]))
           counts.append(len(streak_dict[bucket]))
       else:
           accuracy.append(np.nan)
           counts.append(0)


   # return as a DataFrame.
   return pd.DataFrame({"bucket": buckets, "accuracy": accuracy, "count": counts})




def plot_accuracy_consecutive_condition(
   acc_consecutive, consecutive_condition, on_condition, title="", ax=None, include_handles=False
):
   """ "
   Plots the average accuracy for on_condition trials following a streak of consecutive_condition trials.
   args:
       acc_consecutive: DataFrame with columns 'bucket', 'accuracy', 'count'
       consecutive_condition: the trial type whose consecutive occurrences are counted (e.g. "switch" or "repeat")
       on_condition: the trial type for which accuracy is computed when it follows a streak of consecutive_condition trials
       title: (optional) title for the plot
       ax: (optional) matplotlib axes to plot on
       include_handles: (optional) include custom legend handles for each bucket
   returns:
       ax: matplotlib axes containing the plot
   """
   show_plot = False
   if ax is None:
       ax = plt.gca()
       show_plot = True


   # extract bucket, accuracy, and counts from the DataFrame.
   buckets = acc_consecutive["bucket"]
   accuracy = acc_consecutive["accuracy"]
   counts = acc_consecutive["count"]


   # plot average accuracy per bucket without a legend label.
   (line,) = ax.plot(buckets, accuracy, marker="o", linestyle="-", label=on_condition)
   # chance level line.
   chance_line = ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, linewidth=0.5)


   ax.set_xlabel(f"Consecutive '{consecutive_condition}' trials")
   ax.set_ylabel(f"Accuracy on following trial")
   ax.set_title(title)
   ax.set_xticks(buckets)


   if include_handles:
       # custom legend handles for each bucket, (workaround)
       bucket_handles = []
       for b, cnt in zip(buckets, counts):
           bucket_handles.append(
               Line2D([], [], marker="o", color="C0", linestyle="None", markersize=8, label=f"following {b} (n={cnt})")
           )
       # a legend that includes the bucket handles and the chance line.
       handles = bucket_handles + [chance_line]
       ax.legend(handles=handles, fontsize=10)


   if show_plot:
       plt.draw()


   return ax


