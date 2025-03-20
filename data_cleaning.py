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
    """
    Function to clean the raw data from the experiment.
    The function will clean the data and return a cleaned DataFrame.
    args:
    df_raw: pandas DataFrame
    returns:
    df: pandas DataFrame
    """
    # rename columns for consistency
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
    # take a path to a directory with the raw data files and a directory to save the cleaned data files
    import os
    import glob
    import sys
    import json
    import argparse

    RAW_DATA_DIR = "data/"
    CLEANED_DATA_DIR = "cleaned_data/"

    parser = argparse.ArgumentParser(description="Clean the raw data files from the experiment.")
    parser.add_argument("raw_data_dir", type=str, help="Path to the directory containing the raw data files.")
    parser.add_argument("clean_data_dir", type=str, help="Path to the directory to save the cleaned data files.")
    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    clean_data_dir = args.clean_data_dir

    if raw_data_dir is None:
        raw_data_dir = RAW_DATA_DIR
    if clean_data_dir is None:
        clean_data_dir = CLEANED_DATA_DIR

    if not os.path.exists(raw_data_dir):
        sys.exit(f"Error: Directory '{raw_data_dir}' does not exist.")
    os.makedirs(clean_data_dir, exist_ok=True)

    # get all raw data files
    raw_data_files = glob.glob(os.path.join(raw_data_dir, "*.json"))

    reset_cleaned_data = False  # set to True to delete all cleaned in cleaned_data_dir and re-process the raw data

    # check the number of .jsin files in the raw_data_dir and .csv files in the cleaned_data_dir
    raw_data_names = glob.glob(os.path.join(raw_data_dir, "*.json"))
    cleaned_data_names = glob.glob(os.path.join(clean_data_dir, "*.csv"))

    num_raw_data_files = len(raw_data_names)
    num_cleaned_data_files = len(cleaned_data_names)

    print(f"number of raw data files: {num_raw_data_files}")
    print(f"number of cleaned data files: {num_cleaned_data_files}")

    # list of data files that are in the raw data directory but not in the cleaned data directory
    data_files_to_clean = [
        file for file in raw_data_names if os.path.basename(file).replace(".json", ".csv") not in cleaned_data_names
    ]
    # print("data to clean: ", *data_files_to_clean, sep='\n')
    # clean raw data files
    for data_file in data_files_to_clean:
        with open(data_file, "r") as f:
            data = json.load(f)
        # save to csv only if the data is not already saved
        if not os.path.exists(os.path.join(clean_data_dir, os.path.basename(data_file).replace(".json", ".csv"))):
            df_raw = pd.json_normalize(data)
            df_clean = clean_data(df_raw)
            df_clean.to_csv(
                os.path.join(clean_data_dir, os.path.basename(data_file).replace(".json", ".csv")), index=False
            )

    # get the names of the cleaned data files again after possible upadting
    cleaned_data_names = glob.glob(os.path.join(clean_data_dir, "*.csv"))
    print(f"updated number of cleaned data files: {len(cleaned_data_names)}")
    print("file names: ", *cleaned_data_names, sep="\n")
    # load the cleaned data into dataframes
    data_dfs = [pd.read_csv(data_file) for data_file in cleaned_data_names]

    if len(data_dfs) == 0:
        raise ValueError(
            f"No data to fit files found. Please upload raw data files to: {clean_data_dir}, or cleaned data to: {clean_data_dir}"
        )
