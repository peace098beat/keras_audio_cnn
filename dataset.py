#! coding:utf-8
"""
dataset.py

Created by 0160929 on 2017/07/24 8:59
"""
from pathlib import Path

import os
import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))

NoSairen_dirpath = Path(root_dir).joinpath("sairen+magna").joinpath("NoSairen")
OnSairen_dirpath = Path(root_dir).joinpath("sairen+magna").joinpath("OnSairen")
dataset_filename_X = Path(root_dir).joinpath("sairen+magna").joinpath("sairen-datasets-X")
dataset_filename_y = Path(root_dir).joinpath("sairen+magna").joinpath( "sairen-datasets-y")


def mp3_to_array(file):
    from pydub import AudioSegment

    print("load :"+file)
    # MP3からRAWへの変換
    song = AudioSegment.from_mp3(file)

    # RAWからbytestring型への変換
    song_data = song._data

    # bytestringからNumpy配列への変換
    song_arr = np.fromstring(song_data, np.int16)

    return song_arr


def load(n=1000):

    npy_X = dataset_filename_X.as_posix()+"-n"+str(n)+".npy"
    npy_y = dataset_filename_y.as_posix()+"-n"+str(n)+".npy"

    print(npy_X)
    print(npy_y)

    if(Path(npy_X).exists() and Path(npy_y).exists()):
        print("Exists Dataset")
        X = np.load(npy_X)
        y = np.load(npy_y)
    else:
        print("Not Exists Dataset")
        X,y = _load(n,yIsOneHot=True)
        np.save(npy_X, X)
        np.save(npy_y, y)
    return X,y


def _load(n=10000, yIsOneHot=True):

    TOTAL_SIZE = int(n/2)

    NoSairen_mp3_files = list(NoSairen_dirpath.glob("*.mp3"))
    NoSairen_mp3_files = NoSairen_mp3_files[:TOTAL_SIZE]
    assert len(NoSairen_mp3_files) == n / 2

    OnSairen_mp3_files = list(OnSairen_dirpath.glob("*.mp3"))
    OnSairen_mp3_files = OnSairen_mp3_files[:TOTAL_SIZE]
    assert len(OnSairen_mp3_files) == n / 2


    N1 = len(NoSairen_mp3_files)
    N2 = len(OnSairen_mp3_files)
    N = N1+N2
    print("NoSairen Number =>" + str(N1))
    print("OnSairen Number =>" + str(N2))
    print("Total Number =>" + str(N))


    X1_list=list()
    for file in NoSairen_mp3_files :
        if file.exists():
            try:
                X1_list.append( mp3_to_array(file.as_posix()))
            except Exception as e:
                print(e)

    X2_list = list()
    for  file in OnSairen_mp3_files:
        if file.exists():
            try:
                X2_list.append(mp3_to_array(file.as_posix()))
            except Exception as e:
                print(e)

    X_list = X1_list+X2_list
    X = np.array(X_list)

    assert X.shape[0] == N
    X = X.reshape(X.shape[0], X.shape[1], 1) # samplenum, featurelen, ch

    if(yIsOneHot):
        y = np.array([[1,0]]*N1 + [[0,1]]*N2)
    else:
        y = np.array([0] * N1 + [1] * N2)

    print(y.shape)
    return  X,y

def savenp(savefile):
    pass

if __name__ == '__main__':
    load()
