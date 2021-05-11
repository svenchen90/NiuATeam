import numpy as np

def std_scale(std_data, target_data):
    mean = std_data.mean()
    std = std_data.std()
    # size = std_data.shape[0]
    return (target_data - mean) / std

def std_scale_back(std_data, scaled_data):
    mean = std_data.mean()
    std = std_data.std()
    return scaled_data*std + mean