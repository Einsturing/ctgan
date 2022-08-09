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
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,f1_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

sys.path.append('./utils/')
from visualize import plot_2d, plot_2d_space, plot_3d_space, plot_2d_TF

test_ratio = 0.2
skew_ratio = .04
total_count = 2000
bd = 5.5

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def ball(nums):
    X = np.array([0,0,0]).reshape(-1,3)
    for i in range(nums):
        x = random.random()*2-1
        y = random.random() * 2 - 1
        z = random.random() * 2 - 1

        if x*x +y*y +z*z <=1:
            X = np.append(X,np.array([x,y,z]).reshape(-1,3),axis=0)
    return X

def plot_3d(X,y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ["red","blue"]
    for l,c in zip(np.unique(y),colors):
        ax.scatter(X[y==l, 0], X[y==l, 1], X[y==l, 2], s=1, c=c,label=l)
    plt.show()

def plot_3d_scan(X,y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ["red","blue"]
    for l,c in zip(np.unique(y),colors):
        ax.scatter(X[y==l, 0], X[y==l, 1], X[y==l, 2], s=1, c=c,label=l)
    plt.xlim((-5,5))
    plt.ylim((-5,5))
    plt.show()

def makeBlobs(num = total_count, sr = 0.1):
    X = ball(num) * 5
    y = np.zeros(X.shape[0])
    X_total = X.copy()
    y_total = y.copy()

    X1 = X + [bd,bd,bd]
    y1 = np.zeros(X.shape[0])
    X_total = np.concatenate((X_total,X1),axis=0)
    y_total = np.concatenate((y_total,y1),axis=0)

    X2 = X + [bd,bd,-bd]
    y2 = np.zeros(X.shape[0])
    X_total = np.concatenate((X_total, X2), axis=0)
    y_total = np.concatenate((y_total, y2), axis=0)

    X3 = X + [bd,-bd,bd]
    y3 = np.zeros(X.shape[0])
    X_total = np.concatenate((X_total, X3), axis=0)
    y_total = np.concatenate((y_total, y3), axis=0)

    X4 = X + [bd,-bd,-bd]
    y4 = np.zeros(X.shape[0])
    X_total = np.concatenate((X_total, X4), axis=0)
    y_total = np.concatenate((y_total, y4), axis=0)

    X5 = X + [-bd,bd,bd]
    y5 = np.zeros(X.shape[0])
    X_total = np.concatenate((X_total, X5), axis=0)
    y_total = np.concatenate((y_total, y5), axis=0)

    X6 = X + [-bd,-bd,bd]
    y6 = np.zeros(X.shape[0])
    X_total = np.concatenate((X_total, X6), axis=0)
    y_total = np.concatenate((y_total, y6), axis=0)

    X7 = X + [-bd,bd,-bd]
    y7 = np.zeros(X.shape[0])
    X_total = np.concatenate((X_total, X7), axis=0)
    y_total = np.concatenate((y_total, y7), axis=0)

    X8 = X + [-bd,-bd,-bd]
    y8 = np.zeros(X.shape[0])
    X_total = np.concatenate((X_total, X8), axis=0)
    y_total = np.concatenate((y_total, y8), axis=0)

    X = ball(int(num*sr)) * 5
    y = np.ones(X.shape[0])

    X10 = X + [bd, 0, 0]
    y10 = np.ones(X.shape[0])
    X_total = np.concatenate((X_total, X10), axis=0)
    y_total = np.concatenate((y_total, y10), axis=0)

    X11 = X + [-bd, 0, 0]
    y11 = np.ones(X.shape[0])
    X_total = np.concatenate((X_total, X11), axis=0)
    y_total = np.concatenate((y_total, y11), axis=0)

    X12 = X + [0, bd, 0]
    y12 = np.ones(X.shape[0])
    X_total = np.concatenate((X_total, X12), axis=0)
    y_total = np.concatenate((y_total, y12), axis=0)

    X13 = X + [0, -bd, 0]
    y13 = np.ones(X.shape[0])
    X_total = np.concatenate((X_total, X13), axis=0)
    y_total = np.concatenate((y_total, y13), axis=0)

    X14 = X + [0, 0, bd]
    y14 = np.ones(X.shape[0])
    X_total = np.concatenate((X_total, X14), axis=0)
    y_total = np.concatenate((y_total, y14), axis=0)

    X15 = X + [0, 0, -bd]
    y15 = np.ones(X.shape[0])
    X_total = np.concatenate((X_total, X15), axis=0)
    y_total = np.concatenate((y_total, y15), axis=0)
    plot_3d(X_total,y_total)

    return X_total,y_total

def sidePrecicion(y_test, y_pred,y_pre_pro):
    print(len([y_test for y_test in y_test if y_test == 1]), ":", len(y_test))
    print(len([y_pred for y_pred in y_pred if y_pred == 1]), ":", len(y_pred))
    y_merg = y_pred * y_test
    print(len([y_merg for y_merg in y_merg if y_merg == 1]), ":", len(y_merg))
    print("auc:",roc_auc_score(y_test, y_pred))
    print("auc-pro",roc_auc_score(y_test, y_pre_pro))

def eval(y_test, y_pred):
    print(" Accuracy:", metrics.accuracy_score(y_test, y_pred), " f1-macro: ",
          f1_score(y_test, y_pred, average='macro'))
    print('pricision-macro:', metrics.precision_score(y_test, y_pred, average='macro'), 'recall-macro:',
          metrics.recall_score(y_test, y_pred, average='macro'))
    print('pricision:', metrics.precision_score(y_test, y_pred), 'recall:',
          metrics.recall_score(y_test, y_pred)," f1: ",f1_score(y_test, y_pred))

def gbClassifier_gen(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pre_pro = clf.predict_proba(X_test)[:, 1]

    print("GB:")
    eval(y_test,y_pred)
    sidePrecicion(y_test, y_pred,y_pre_pro)


if __name__ == '__main__':
    path = "Results/3d_mix_complex/"
    X,y = makeBlobs(num=total_count,sr=0.2)

    array = X[:, 2]
    values = np.linspace(np.min(array),np.max(array),num=20)
    for i in range(np.shape(X)[0]):
        X[i,2]= find_nearest(values,X[i,2])

    plot_3d(X, y)
    plt.figure()
    plt.scatter(X[:,1],X[:,2],s=0.2)
    plt.show()

    X_maj = X[np.where(y == 0)]
    y_maj = y[np.where(y == 0)]
    X_min = X[np.where(y == 1)]
    y_min = y[np.where(y == 1)]
    plt.figure()
    plt.scatter(X_min[:, 0], X_min[:, 1], s=0.1)
    plt.title("X_MIN-0-1")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()

    plt.figure()
    plt.scatter(X_min[:, 0], X_min[:, 2], s=0.1)
    plt.title("X_MIN-0-2")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()

    plt.figure()
    plt.scatter(X_min[:, 1], X_min[:, 2], s=0.1)
    plt.title("X_MIN-1-2")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
    np.savetxt(path + "X_train.csv",X_train,delimiter=",")
    np.savetxt(path + "y_train.csv",y_train,delimiter=",")
    np.savetxt(path + "X_test.csv",X_test,delimiter=",")
    np.savetxt(path + "y_test.csv",y_test,delimiter=",")
    gbClassifier_gen(X_train, X_test, y_train, y_test)
