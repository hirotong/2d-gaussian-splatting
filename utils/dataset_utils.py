import pickle
import numpy as np

def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)