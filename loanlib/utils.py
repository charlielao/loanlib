import numpy as np


def shift_by_months(arr: np.array, number_of_months: int = 1, padded_value: float = 0.0):
    return np.concatenate((np.zeros(number_of_months), np.roll(arr, number_of_months)[number_of_months:]))


def get_first_truth_value(arr: np.array):
    idx = np.argmax(arr)
    return idx if arr[idx] else -1


def fill_static(fill_value, size: int, is_date: bool = True):
    return np.full(shape=size, fill_value=fill_value if fill_value is None else (np.datetime64(fill_value) if is_date else fill_value))
