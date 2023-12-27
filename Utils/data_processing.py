import pandas as pd
import numpy as np
import random
import gc
import time
import shutil
import re
import os
from tqdm import tqdm
import glob
from parameters import HyperParameters
from utils import DataSplitter

parameters  = HyperParameters()

# ✨  Gather the knowledge from JSON files ✨
def read_json_data(mode="train"):
    """
    Explores a collection of JSON files, carefully extracting their valuable insights. 

    Args:
        mode (str, optional): Specifies the mode of operation ("train" or "test"). Defaults to "train".

    Returns:
        pd.DataFrame: A meticulously organized DataFrame containing the extracted knowledge.
    """

    path_to_json_realm = sorted(list(glob.glob(parameters.data_path + mode + "/*.json".format(mode))))  # ️ Map out the path to the JSON files
    knowledge_vault = pd.concat(
        [
            pd.read_json(path, dtype={"cell_type": str, "source": str})  #  Decode each JSON file
                .assign(id=path.split("/")[-1].split(".")[0])  #  Assign a unique identifier
                .rename_axis("cell_id")  # Rename for clarity
            for path in tqdm(path_to_json_realm)  # ✨ Progress tracker: Watch the knowledge accumulate!
        ]
    ).reset_index(drop=False)  # Refine the structure
    return knowledge_vault[["id", "cell_id", "cell_type", "source"]]  #  Focus on essential columns


#  Meticulously clean and prepare text data 
def preprocess_text(text_data):
    """
    Perform a series of text preprocessing steps to ensure data consistency and suitability for analysis.

    Args:
        text_data (str): The raw text to be preprocessed.

    Returns:
        str: The clean and prepared text.
    """

    #  ⚔️⚔️⚔️ Eliminate unwanted characters ⚔️⚔️⚔️
    text_data = re.sub(r'\W', ' ', str(text_data))  # Remove non-alphanumeric characters
    text_data = text_data.replace('_', ' ')  # Treat underscores as spaces

    #   Isolate single letters surrounded by spaces 
    text_data = re.sub(r'\s+[a-zA-Z]\s+', ' ', text_data)  # Target them for removal

    #  ️️️ Collapse excess whitespace ️️️
    text_data = re.sub(r'\s+', ' ', text_data, flags=re.I)  # Ensure uniform spacing

    #  ⬇️⬇️⬇️ Unify case for consistency ⬇️⬇️⬇️
    text_data = text_data.lower()  # Convert all characters to lowercase

    return text_data  # Return the polished text


# ️  Prepare the knowledge for deeper understanding ️
def preprocess_df(df):
    """
    Meticulously organizes and refines the extracted knowledge, preparing it for further analysis.

    Args:
        df (pd.DataFrame): The DataFrame containing the raw knowledge.

    Returns:
        pd.DataFrame: A transformed DataFrame, ready for deeper insights.
    """

    #  Count the cells within each unit of knowledge
    df["cell_count"] = df.groupby(by=["id"])["cell_id"].transform("count")

    # ️ Categorize cell types for efficient understanding
    df["cell_type"] = df["cell_type"].map({"code": 0, "markdown": 1}).fillna(0).astype(int)

    #  Calculate the distribution of knowledge types
    df["markdown_count"] = df.groupby(by=["id"])["cell_type"].transform("sum")
    df["code_count"] = df["cell_count"] - df["markdown_count"]

    # ⚖️ Normalize importance scores for a balanced perspective
    df["rank"] = df["rank"] / df["cell_count"]

    # ✨✨✨ Text preprocessing for clarity and precision ✨✨✨
    df["source"] = df["source"].apply(lambda x: x.lower().strip())  #  Sweep away capitalization and extra spaces
    df["source"] = df["source"].apply(lambda x: preprocess_text(x))  # Apply specialized text cleaning techniques
    df["source"] = df["source"].str.replace("[SEP]", "")  # Remove unnecessary separators
    df["source"] = df["source"].str.replace("[CLS]", "")  # Eliminate extraneous markers
    df["source"] = df["source"].apply(lambda x: re.sub(" +", " ", x))  # Condense extra spaces for compactness

    return df  #  Return the refined knowledge, ready for exploration!


def get_data(seed=42, mode=0):
    if os.path.exists("../input/trainpicklefile/train_df.pkl"):
        train_df = pd.read_pickle("../input/trainpicklefile/train_df.pkl")
    else:
        train_df = read_json_data(mode="train")
        train_orders = pd.read_csv("../input/" + "train_orders.csv")
        train_ancestors = pd.read_csv("../input/" + "train_ancestors.csv")
        train_orders["cell_id"] = train_orders["cell_order"].str.split()
        train_orders = train_orders.explode(column="cell_id")
        train_orders["flag"] = range(len(train_orders))
        train_orders["rank"] = train_orders.groupby(by=["id"])["flag"].rank(ascending=True, method="first").astype(int)
        del train_orders["flag"], train_orders["cell_order"]
        print(train_orders)
        train_df = train_df.merge(train_orders, on=["id", "cell_id"], how="left")
        train_df = train_df.merge(train_ancestors[["id", "ancestor_id"]], on=["id"], how="left")
        train_df.to_pickle("train_df.pkl")
    train_df = DataSplitter(seed, parameters.k_folds).group_split(train_df, group_col="ancestor_id")
    train_df = preprocess_df(train_df)
    train_df = pd.concat([train_df[train_df["cell_type"] == 0], train_df[train_df["cell_type"] == 1].sample(frac=1.0)]).reset_index(drop=True)


#   Building a well-structured DataFrame 
def get_truncated_df(df, cell_count=128, id_col='id2', group_col='id', max_random_cnt=100, expand_ratio=5):
    """
    Crafts a meticulously crafted DataFrame by strategically truncating and expanding data as needed.
    This method ensures that the resulting DataFrame adheres to specific cell count constraints while
    preserving valuable information for further analysis.

    Args:
        df (pd.DataFrame): The original DataFrame containing the data to be organized.
        cell_count (int, optional): The maximum desired number of cells per group. Defaults to 128.
        id_col (str, optional): The name of the column to use for assigning unique identifiers. Defaults to 'id2'.
        group_col (str, optional): The name of the column to use for grouping data. Defaults to 'id'.
        max_random_cnt (int, optional): The maximum number of random samples to generate for expansion. Defaults to 100.
        expand_ratio (int, optional): The factor by which to expand the DataFrame when necessary. Defaults to 5.

    Returns:
        pd.DataFrame: The meticulously constructed DataFrame, ready for further exploration.
    """

    #  🪜 Separating straightforward cases 🪜
    tmp1 = df[df['cell_count'] <= cell_count].reset_index(drop=True)  # Data already within limits
    tmp1.loc[:, id_col] = 1  # Assigning initial identifiers
    tmp2 = df[df['cell_count'] > cell_count].reset_index(drop=True)  # Data requiring special attention

    #  ✨✨✨ Initiating the DataFrame construction journey ✨✨✨
    res = [tmp1]  # Laying the foundation

    for _, df_g in tmp2.groupby(by=group_col):
        #   Shuffling the data for a touch of surprise 
        df_g = df_g.sample(frac=1.0).reset_index(drop=True)

        step = min(cell_count // 2, len(df_g) - cell_count)  # Determining a suitable step size
        step = max(step, 1)  # Ensuring progress

        id_col_count = 1  # Preparing unique identifiers

        for i in range(0, len(df_g), step):
            res_tmp = df_g.iloc[i:i + cell_count]  # Capturing a segment
            if len(res_tmp) != cell_count:  # Handling edge cases gracefully
                res_tmp = df_g.iloc[-cell_count:]
            res_tmp.loc[:, id_col] = id_col_count  # Assigning identifiers for clarity
            id_col_count += 1  # Ready for the next segment
            res.append(res_tmp)  # Merging into the masterpiece

            if i + cell_count >= len(df_g):  # Recognizing completion
                break  # A moment of pause

        #   Expanding horizons when needed 
        if len(df_g) // cell_count > 1.3:
            random_cnt = int(len(df_g) // cell_count * expand_ratio)  # Unleashing the potential
            random_cnt = min(random_cnt, max_random_cnt)  # Maintaining balance

            for i in range(random_cnt):  # Embracing diversity
                res_tmp = df_g.sample(n=cell_count).reset_index(drop=True)  # Creating new perspectives
                res_tmp.loc[:, id_col] = id_col_count  # Assigning unique identities
                id_col_count += 1  # Ever evolving
                res.append(res_tmp)  # Weaving together the tapestry

    #  ✨✨✨ Finalizing the masterpiece ✨✨✨
    res = pd.concat(res).reset_index(drop=True)  # Bringing it all together
    res = res.sort_values(by=['id', id_col, 'cell_type', 'rank'], ascending=True)  # Ensuring order and clarity
    res = res.groupby(by=['id', id_col, 'fold_flag', 'cell_count'], as_index=False, sort=False)[
        ['cell_id', 'cell_type', 'source', 'rank']].agg(list)  # Grouping for efficiency
    return res  # The meticulously crafted DataFrame, ready for further exploration.