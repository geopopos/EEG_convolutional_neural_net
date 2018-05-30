import numpy as np

from scipy.io import loadmat

import json

def load_data(file):
    with open(file, "r") as raw_data:
        pp_data = json.loads(raw_data.read())
    return pp_data

def save_data(file, data):
    keys = ["X_train", "Y_train", "X_test", "Y_test"]
    for key in keys:
        data[key] = data[key].tolist()
    with open(file, "w") as json_file:
        json.dump(data, json_file)
