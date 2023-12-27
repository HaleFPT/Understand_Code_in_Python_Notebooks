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

def read_json_data(mode="train"):
    path_train = sorted(list(glob.glob(parameters.data_path + mode + "/*.json".format(mode))))
    res = pd.concat(
        [pd.read_json(path, dtype={"cell_type": str, "source": str}).assign(
            id=path.split("/")[-1].split(".")[0]).rename_axis("cell_id") 
                for path in tqdm(path_train)]).reset_index(drop=False)
    res = res[["id", "cell_id", "cell_type", "source"]]
    return res