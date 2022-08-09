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
from sklearn.neighbors import NearestNeighbors


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

X, y, X_test, y_test = makeBlobs(split=True, sr=0.1)
np.savetxt("Results/2d_base/x.csv", X, delimiter=",")
np.savetxt("Results/2d_base/y.csv", y, delimiter=",")
np.savetxt("Results/2d_base/x_test.csv", X_test, delimiter=",")
np.savetxt("Results/2d_base/y_test.csv", y_test, delimiter=",")

# array = X[:,1]
# values = np.linspace(np.min(array),np.max(array),num=20)
# for i in range(np.shape(X)[0]):
#     X[i,1]= find_nearest(values,X[i,1])
#
#
# X_maj = X[np.where(y==0)]
# y_maj = y[np.where(y==0)]
# X_min = X[np.where(y==1)]
# y_min = y[np.where(y==1)]


# for i in range(10):
#     X = np.concatenate((X,X_min),axis=0)
#     y = np.concatenate((y,y_min),axis=0)
# print(np.shape(X),np.shape(X_test))
# plt.scatter(X_maj[:,0],X_maj[:,1],s=.1)
# plt.show()
# plt.scatter(X_min[:,0],X_min[:,1],s=.1)
# plt.show()
#
#
# neigh = NearestNeighbors(radius=0.3)
# neigh.fit(X_min)
# rng = neigh.radius_neighbors_graph(X_min).toarray()
#
# indices = np.sum(rng,axis=1)
# print(np.mean(indices))
# idx= np.where(indices<5) and np.where(indices>2)
#
# for i in range(5):
#     noises = np.concatenate((np.random.normal(loc=0,scale=1,size=(len(idx),1)),np.zeros((len(idx),1))),axis=1)
#     X_aug = X_min[idx] + noises
#     X_min = np.append(X_min,X_aug,axis=0)

data = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))))
data.columns = ["f1", "f2", "y"]

# data = load_demo()
print(data.columns)
print(type(data.shape))


# Names of the columns that are discrete
discrete_columns = [
    'y'
]

ctgan = CTGANSynthesizer(epochs=1, verbose=True)
ctgan.fit(data, discrete_columns)


# Synthetic copy
samples_ori = ctgan.sample(20000)
samples = samples_ori.values[:, :2]
labels = samples_ori.values[:, 2]
np.savetxt("Results/samples_2d.csv", samples_ori, delimiter=",")

samples_maj = samples[np.where(labels == 0)]
labels_maj = labels[np.where(labels == 0)]
samples_min = samples[np.where(labels == 1)]
labels_min = labels[np.where(labels == 1)]

print(np.shape(samples_maj), np.shape(samples_min))

plt.scatter(samples_maj[:, 0], samples_maj[:, 1], s=.1)
plt.show()
plt.scatter(samples_min[:, 0], samples_min[:, 1], s=.1)
plt.show()


