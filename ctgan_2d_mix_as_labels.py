# -*- coding: utf-8 -*-
# @Author  : Yandai
# @Email   : 18146573209@163.com
# @File    : ctgan_base.py
# @Time    : 2021-7-20 13:50
# @Software: PyCharm

from ctgan import CTGANSynthesizer
from ctgan import load_demo
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gen_blobs import makeBlobs
from random import sample


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


X = np.loadtxt("Results/ct_2d_mix_as_labels/Origins/x.csv", delimiter=",")
y = np.loadtxt("Results/ct_2d_mix_as_labels/Origins/y.csv", delimiter=",")
X_test = np.loadtxt("Results/ct_2d_mix_as_labels/Origins/x_test.csv", delimiter=",")
y_test = np.loadtxt("Results/ct_2d_mix_as_labels/Origins/y_test.csv", delimiter=",")

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=0.1)
plt.title("X_mixed")
plt.show()
plt.clf()

plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], s=0.1)
plt.title("X_test")
plt.show()
plt.clf()

X_maj = X[np.where(y == 0)]
y_maj = y[np.where(y == 0)]
X_min = X[np.where(y == 1)]
y_min = y[np.where(y == 1)]

print(np.shape(X), np.shape(X_test))
plt.figure()
plt.scatter(X_maj[:, 0], X_maj[:, 1], s=.1)
plt.title("Ori-Maj")
plt.savefig("Results/ct_2d_mix_as_labels/Imgs/maj_ori.png")
plt.clf()
plt.figure()
plt.scatter(X_min[:, 0], X_min[:, 1], s=.1)
plt.title("Ori-Min")
plt.savefig("Results/ct_2d_mix_as_labels/Imgs/min_ori.png")
plt.clf()

data = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))))
data.columns = ["f1", "f2", "y"]

# data = load_demo()
print(data.columns)
print(type(data.shape))

# Names of the columns that are discrete
discrete_columns = [
    'f2',
    'y'
]

ctgan = CTGANSynthesizer(epochs=5000, verbose=True)
ctgan.fit(data, discrete_columns)


# Synthetic copy
def Sample(iter):
    samples_ori = ctgan.sample(20000)
    print("samples_ori:", samples_ori.shape)
    samples = samples_ori.values[:, :2]
    labels = samples_ori.values[:, 2]
    np.savetxt("Results/ct_2d_mix_as_labels/Samples/samples_2d_{}.csv".format(iter), samples_ori,
               delimiter=",")

    samples_maj = samples[np.where(labels == 0)]
    labels_maj = labels[np.where(labels == 0)]
    samples_min = samples[np.where(labels == 1)]
    labels_min = labels[np.where(labels == 1)]

    # print(np.shape(samples_maj),np.shape(samples_min))

    plt.scatter(samples_maj[:, 0], samples_maj[:, 1], s=.1)
    plt.title("Samples-Maj")
    plt.savefig("Results/ct_2d_mix_as_labels/Imgs/maj_{}.png".format(iter))
    # plt.show()
    plt.clf()
    plt.scatter(samples_min[:, 0], samples_min[:, 1], s=.1)
    plt.title("Samples-Min")
    plt.savefig("Results/ct_2d_mix_as_labels/Imgs/min_{}.png".format(iter))
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    for i in range(10):
        print(i)
        Sample(i)
