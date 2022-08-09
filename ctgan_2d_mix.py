# -*- coding: utf-8 -*-
# @Author  : Yandai
# @Email   : 18146573209@163.com
# @File    : ctgan_base.py
# @Time    : 2021-7-20 13:50
# @Software: PyCharm
import os.path
from CTGAN.synthesizers.ctgan import CTGANSynthesizer
from CTGAN.synthesizers.base import BaseSynthesizer
import warnings

from synthesizers.data_transformer import DataTransformer

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CTGAN.gen_blobs import makeBlobs


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def init():
    # X, y, X_test, y_test = makeBlobs(split=True, sr=0.1, total_count=100000)
    X = np.loadtxt("Results/2d_mix_balance/Origins/x.csv", delimiter=",")
    y = np.loadtxt("Results/2d_mix_balance/Origins/y.csv", delimiter=",")
    # active = np.random.randint(0, 2, len(X))
    # array = X[:, 1]
    # array_test = X_test[:, 1]
    # dis = np.max(array) - np.min(array)
    # X[:, 1][(X[:, 1] <= np.min(array) + dis * .2)] = np.min(array)
    # X[:, 1][(X[:, 1] >= np.max(array) - dis * .2)] = np.max(array)

    # values = np.linspace(np.min(array), np.max(array), num=20)
    # values_test = np.linspace(np.min(array_test), np.max(array_test), num=20)
    # for i in range(np.shape(X)[0]):
    #     X[i, 1] = find_nearest(values, X[i, 1])
    #
    # for i in range(np.shape(X_test)[0]):
    #     X_test[i, 1] = find_nearest(values_test, X_test[i, 1])

    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], s=0.1)
    # plt.title("X_mixed")
    # plt.show()

    # plt.figure()
    # plt.scatter(X_test[:, 0], X_test[:, 1], s=0.1)
    # plt.title("X_test")

    X_maj = X[np.where(y == 0)]
    y_maj = y[np.where(y == 0)]
    X_min = X[np.where(y == 1)]
    y_min = y[np.where(y == 1)]

    # X_maj = X_maj[:300000]
    # y_maj = y_maj[:300000]
    # X_min = X_min[:30000]
    # y_min = y_min[:30000]

    # x = np.concatenate((X_maj, X_min), axis=0)
    # y = np.concatenate((y_maj, y_min), axis=0)

    # np.savetxt("Results/2d_mix_balance/Origins/x.csv", x, delimiter=",")
    # np.savetxt("Results/2d_mix_balance/Origins/y.csv", y, delimiter=",")
    # np.savetxt("Results/2d_mix_balance/Origins/x_test.csv", X_test, delimiter=",")
    # np.savetxt("Results/2d_mix_balance/Origins/y_test.csv", y_test, delimiter=",")

    # print(np.shape(X), np.shape(X_test))
    # plt.figure()
    # plt.scatter(X_maj[:, 0], X_maj[:, 1], s=.1)
    # plt.title("Ori-Maj")
    # plt.savefig("Results/2d_mix_balance/Imgs/maj_ori.png")
    # plt.show()
    # plt.clf()
    # plt.figure()
    # plt.scatter(X_min[:, 0], X_min[:, 1], s=.1)
    # plt.title("Ori-Min")
    # plt.savefig("Results/2d_mix_balance/Imgs/min_ori.png")
    # plt.show()
    # plt.clf()

    # data = pd.read_csv("Results/adult/Origins/adult-0.05.csv")
    # data = pd.read_csv("Results/scene/Origins/scene.csv")
    # data = pd.read_csv("Results/wine_quality/Origins/wine_quality.csv")[:-400]
    data = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))))
    data.columns = ["f1", "f2", "y"]
    # data.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    #                 "f13", "f14", "y"]

    print(data.columns)
    print(type(data.shape))

    # Names of the columns that are discrete
    discrete_columns = ['f2', 'y']
    # discrete_columns = ['f1', 'f2', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13',
    #                     'y']
    # discrete_columns = ['y']

    ctgan = CTGANSynthesizer(verbose=True, batch_size=500)
    ctgan.fit(data, 5000, discrete_columns)


def importdata():
    gtest = pd.read_csv('Results/2d_mix_balance/Origins/x_test.csv', header=None)
    glabs = pd.read_csv('Results/2d_mix_balance/Origins/y_test.csv', header=None)
    vdata = pd.DataFrame(
        np.hstack((gtest.values, glabs.values.reshape(-1, 1))).astype("float32")).iloc[:1000, :]
    vdata.columns = ["f1", "f2", "y"]
    # discrete_columns = ['f2', 'y']
    # transformer = DataTransformer()
    # transformer.fit(data, discrete_columns)
    # vdata = transformer.transform(data)

    gtest = pd.read_csv('Results/2d_mix_balance/Origins/x.csv', header=None)
    glabs = pd.read_csv('Results/2d_mix_balance/Origins/y.csv', header=None)
    tdata = pd.DataFrame(
        np.hstack((gtest.values, glabs.values.reshape(-1, 1))).astype("float32")).iloc[:100, :]
    tdata.columns = ["f1", "f2", "y"]
    # transformer = DataTransformer()
    # transformer.fit(data, discrete_columns)
    # tdata = transformer.transform(data)
    # print(tdata[:10, :1])
    # np.random.shuffle(tdata)
    # print(tdata[:10, :1])
    return tdata, vdata


def load():
    ctgan = BaseSynthesizer()
    ctgan = ctgan.load(os.path.join('./5000_stable_auc_model.pt'))

    ctgan.auc_fit(2500, 500)


# Synthetic copy
def Sample(iter):
    ctgan = BaseSynthesizer()
    ctgan = ctgan.load(os.path.join('./{}_auc_model.pt'.format(iter)))

    samples_ori = ctgan.sample(20000)
    print("samples_ori:", samples_ori.shape)
    samples_ori = samples_ori.values
    # samples_ori = samples_ori[samples_ori[:, -1] == 1, :-1]
    samples = samples_ori[:, :-1]
    labels = samples_ori[:, -1]
    np.savetxt("Results/2d_mix_balance/Samples/{}_auc_samples_2d.csv".format(iter), samples_ori,
               delimiter=",")

    samples_maj = samples[np.where(labels == 0)]
    labels_maj = labels[np.where(labels == 0)]
    samples_min = samples[np.where(labels == 1)]
    labels_min = labels[np.where(labels == 1)]
    #
    # # print(np.shape(samples_maj),np.shape(samples_min))
    #
    plt.scatter(samples_maj[:, 0], samples_maj[:, 1], s=.1)
    plt.title("Samples-Maj")
    plt.savefig("Results/2d_mix_balance/Imgs/{}_auc_maj.png".format(iter))
    plt.show()
    plt.clf()
    plt.scatter(samples_min[:, 0], samples_min[:, 1], s=.1)
    plt.title("Samples-Min")
    plt.savefig("Results/2d_mix_balance/Imgs/{}_auc_min.png".format(iter))
    plt.show()
    plt.clf()


if __name__ == '__main__':
    # init()
    load()
    # Sample(2500)
