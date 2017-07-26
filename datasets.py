#! coding:utf-8
"""
datasets.py

Created by 0160929 on 2017/07/25 22:21
"""
import os
from pathlib import Path

import numpy as np

Root_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def _load(NoSairenPath, OnSairenPath, yIsOneHot=True):
    NoSairenArray = np.load(NoSairenPath.as_posix())
    OnSairenArray = np.load(OnSairenPath.as_posix())

    N1 = NoSairenArray.shape[0]
    N2 = OnSairenArray.shape[0]
    assert N1 == N2

    X = np.concatenate((NoSairenArray, OnSairenArray), axis=0)
    assert (N1 + N2) == X.shape[0]

    if (yIsOneHot):
        y = np.array([[1, 0]] * N1 + [[0, 1]] * N2)
    else:
        y = np.array([0] * N1 + [1] * N2)

    assert X.shape[0] == y.shape[0]

    return X, y


def get_mell(dataset_dir, num=1000, yIsOneHot=True):
    npy_dir = Path(dataset_dir)
    NoSairenPath = npy_dir.joinpath("NoSairen_mell_" + str(num) + ".npy")
    OnSairenPath = npy_dir.joinpath("OnSairen_mell_" + str(num) + ".npy")
    X, y = _load(NoSairenPath, OnSairenPath, yIsOneHot)
    return X,y

def get_mfcc(dataset_dir, num=1000, yIsOneHot=True):
    npy_dir = Path(dataset_dir)
    NoSairenPath = npy_dir.joinpath("NoSairen_mfcc_" + str(num) + ".npy")
    OnSairenPath = npy_dir.joinpath("OnSairen_mfcc_" + str(num) + ".npy")
    X, y = _load(NoSairenPath, OnSairenPath, yIsOneHot)
    return X, y


if __name__ == '__main__':
    get_mell(Path(__file__).parent.joinpath("data-30dBF"), num=100)
    print("test success!")
