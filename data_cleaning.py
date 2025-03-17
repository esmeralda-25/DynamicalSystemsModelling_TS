import pandas as pd
import numpy as np
import re
import ast


# clean stimulus from html tags
def clean_stimulus(stimulus):
    if isinstance(stimulus, str):  # Ensure input is a string
        return re.sub(r"<.*?>", "", stimulus).replace("\n", "").strip()
    return stimulus  # Return as-is if not a string


def clean_data(df_raw):
    # Rename columns for consistency
    df = df_raw.rename(columns=lambda x: x.replace("bean_", "")).copy()
    # get only needed columns and drop the rest
    relevant_columns = ["task_type", "rt", "correct", "response", "correct_key", "choices", "text"]
    df = df[relevant_columns]
    # rename columns to fit conventions
    df = df.rename(columns={"response": "participant_response"})
    df = df.rename(columns={"correct": "response", "text": "stimuli", "task_type": "task", "choices": "key_choices"})

    # convert string representations of lists in the 'key_choices' column into actual lists
    df["key_choices"] = df["key_choices"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # filter the DataFrame to keep only rows where 'key_choices' is exactly ['b', 'n'], to filter our instructions
    # df = df[df['key_choices'].apply(lambda x: x == ['b', 'n'])].copy()
    # filter the df for only the relevant task trails, get rid of the instruction and feedback trails
    # where task is either letter_task or number_task
    df = df[df["task"].isin(["letter_task", "number_task"])].copy()
    df = df.reset_index(drop=True)
    df["trial_index"] = df.index

    # renaming task column entries
    df["task"] = df["task"].replace({"letter_task": "letter", "number_task": "number"})

    # apply the clean_stimulus function to the 'stimuli' column
    df["stimuli"] = df["stimuli"].apply(clean_stimulus)

    # get rid of outliers (rt NaN, negative or too long)
    max_rt = 3000
    df = df[(df["rt"] >= 0) & (df["rt"] <= max_rt) & (df["rt"].notna())]
    df = df.dropna()

    # convert rt from ms to seconds
    df["rt"] = df["rt"] / 1000
    # convert boolean columns to numbers (0,1)
    df["response"] = df["response"].astype(int)

    # add a column for trial_type (repeat, switch)
    df["trial_type"] = df["task"].eq(df["task"].shift(1))
    df["trial_type"] = df["trial_type"].replace({True: "repeat", False: "switch"})

    # discard the first trail (trial_types of interest don't apply)
    df = df.iloc[1:]

    # add a alternative (boolean helper) column for trial_type in case needed. as integer
    df["switch"] = df["trial_type"].eq("switch").astype(int)

    # reset index
    df.reset_index(drop=True, inplace=True)

    # calculate running accuracy over trails
    df["accuracy"] = (df["response"].cumsum() / (df.index + 1)).astype(float)

    # adding choice_adjusted
    # for the models the correct response to the letter task is allways considered to be the first choice (0) and the second choice (1) for the number task
    df["choice_adjusted"] = df.apply(
        lambda row: 1 - row["response"] if row["task"] == "letter" else row["response"], axis=1
    ).astype(int)

    # add trail index
    df["trial_index"] = df.index
    # reorder columns for readability
    df = df[
        [
            "task",
            "trial_type",
            "response",
            "accuracy",
            "choice_adjusted",
            "rt",
            "switch",
            "participant_response",
            "correct_key",
            "key_choices",
            "stimuli",
            "trial_index",
        ]
    ]

    print(f" data cleaning done. number of valid trails: {len(df)}")
    return df


if __name__ == "__main__":
    # process all data and save as csv files
    pass
