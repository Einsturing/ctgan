# -*- coding: utf-8 -*-
# @Author  : Yandai
# @Email   : 18146573209@163.com
# @File    : gen_blobs.py
# @Time    : 2021-5-6 18:46
# @Software: PyCharm


import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_gaussian_quantiles, make_blobs
from sklearn.model_selection import train_test_split
import sys

sys.path.append('./utils/')
from visualize import plot_2d, plot_2d_space, plot_3d_space, plot_2d_TF

test_ratio = 0.2
skew_ratio = .1
bound_width = 5


def makeBlobs(split=True, sr=skew_ratio, total_count=1000):
    skew_ratio = sr
    Xg, yg = make_blobs(cluster_std=1.5, center_box=(-0.2, 0.2), centers=[[0, 0]], n_features=2, n_samples=total_count,
                        random_state=None, shuffle=True)
    # print('Xg,yg',np.shape(Xg),np.shape(yg)) 5000
    df = pd.DataFrame(Xg)
    df['y'] = yg
    df['y'] = 0

    Xr = Xg.copy()[:int(total_count * skew_ratio)]
    yr = yg.copy()[:int(total_count * skew_ratio)]

    X = Xr.copy()
    y = yr.copy()
    dup = pd.DataFrame(X)
    dup.iloc[:, 0] += bound_width
    dup['y'] = 1
    df = pd.concat([df, dup], ignore_index=True)

    # df.target.value_counts().plot(kind='bar', title='Count (target)')
    # plot_2d_space(X, y)

    # X, y = make_blobs(cluster_std=1.0, center_box=(-1, 1), centers=[[0, 2 * bound_width]], n_features=2,
    #                   n_samples=total_count, random_state=None)
    X = Xg.copy()
    y = yg.copy()
    dup = pd.DataFrame(X)
    dup.iloc[:, 1] += 2 * bound_width
    dup['y'] = 0
    # print(df)
    # print(dup)
    df = pd.concat([df, dup], ignore_index=True)

    X = Xr.copy()
    y = yr.copy()
    dup = pd.DataFrame(X)
    dup.iloc[:, 1] += bound_width
    dup['y'] = 1
    # print(df)
    # print(dup)
    df = pd.concat([df, dup], ignore_index=True)

    # X, y = make_blobs(cluster_std=1.0, center_box=(-1, 1), centers=[[2 * bound_width, 0]], n_features=2,
    #                   n_samples=total_count, random_state=None)
    X = Xg.copy()
    y = yg.copy()
    dup = pd.DataFrame(X)
    dup.iloc[:, 0] += 2 * bound_width
    dup['y'] = 0
    df = pd.concat([df, dup], ignore_index=True)

    X = Xr.copy()
    y = yr.copy()
    dup = pd.DataFrame(X)
    dup.iloc[:, 0] += 2 * bound_width
    dup.iloc[:, 1] += bound_width
    dup['y'] = 1
    df = pd.concat([df, dup], ignore_index=True)

    # X, y = make_blobs(cluster_std=1.0, center_box=(-1, 1), centers=[[2 * bound_width, 2 * bound_width]], n_features=2,
    #                   n_samples=total_count, random_state=None)
    X = Xg.copy()
    y = yg.copy()
    dup = pd.DataFrame(X)
    dup.iloc[:, 0] += 2 * bound_width
    dup.iloc[:, 1] += 2 * bound_width
    dup['y'] = 0
    df = pd.concat([df, dup], ignore_index=True)

    X = Xr.copy()
    y = yr.copy()
    dup = pd.DataFrame(X)
    dup.iloc[:, 0] += bound_width
    dup.iloc[:, 1] += 2 * bound_width
    dup['y'] = 1
    df = pd.concat([df, dup], ignore_index=True)

    # X, y = make_blobs(cluster_std=1.0, center_box=(-1, 1), centers=[[bound_width, bound_width]], n_features=2,
    #                   n_samples=total_count, random_state=None)
    X = Xg.copy()
    y = yg.copy()
    dup = pd.DataFrame(X)
    dup.iloc[:, 0] += bound_width
    dup.iloc[:, 1] += bound_width
    dup['y'] = 0
    df = pd.concat([df, dup], ignore_index=True)

    # print(df.shape)
    # print(df.iloc[:, 0:2].values)
    if split:
        X_train, X_test, y_train, y_test = train_test_split(
            df.iloc[:, 0:2].values, df.loc[:, 'y'].values, test_size=0.2, random_state=1)
        # fc, fd = ['f1', 'f2'], 'y'
        # df.columns[0:2], df.columns[2]
        # X_train = pd.DataFrame(X_train, columns=fc)
        # y_train = pd.Series(y_train, name=fd)
        # X_test = pd.DataFrame(X_test, columns=fc)
        # y_test = pd.Series(y_test, name=fd)
        return X_train, y_train, X_test, y_test
        # pd.DataFrame(y_train,columns=df.columns[2]),
        # pd.DataFrame(y_test, columns=df.columns[2])
    else:
        return df
