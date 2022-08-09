import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def idx_I0I1(y):
    return ((np.where(y == 0)[0], np.where(y == 1)[0]))


def AUROC(eta, idx0, idx1):
    den = len(idx0) * len(idx1)
    num = 0
    for i in idx1:
        num += sum(eta[i] > eta[idx0]) + 0.5 * sum(eta[i] == eta[idx0])
    return (num / den)


def cAUROC(w, X, idx0, idx1):
    eta = X.dot(w)
    den = len(idx0) * len(idx1)
    num = 0
    for i in idx1:
        num += sum(np.log(sigmoid(eta[i] - eta[idx0])))
    return (- num / den)


def dcAUROC(w, X, idx0, idx1):
    eta = X.dot(w)
    n0, n1 = len(idx0), len(idx1)
    den = n0 * n1
    num = 0
    for i in idx1:
        num += ((1 - sigmoid(eta[i] - eta[idx0])).reshape([n0, 1]) * (X[[i]] - X[idx0])).sum(
            axis=0)  # *
    return (- num / den)


class ffnet(nn.Module):
    def __init__(self, num_features):
        super(ffnet, self).__init__()
        p = num_features
        self.fc1 = nn.Linear(p, 36)
        self.fc2 = nn.Linear(36, 12)
        self.fc3 = nn.Linear(12, 6)
        self.fc4 = nn.Linear(6, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return (x)


if __name__ == '__main__':
    np.random.seed(1234)
    data = fetch_california_housing(download_if_missing=True)
    cn_cali = data.feature_names
    X_cali = data.data
    y_cali = data.target
    y_cali += np.random.randn(y_cali.shape[0]) * (y_cali.std())
    y_cali = np.where(y_cali > np.quantile(y_cali, 0.95), 1, 0)
    y_cali_train, y_cali_test, X_cali_train, X_cali_test = \
        train_test_split(y_cali, X_cali, test_size=0.2, random_state=1234, stratify=y_cali)
    enc = StandardScaler().fit(X_cali_train)
    # Binary loss function
    criterion = nn.BCEWithLogitsLoss()
    # Seed the network
    torch.manual_seed(1234)
    nnet = ffnet(num_features=X_cali.shape[1])
    optimizer = torch.optim.Adam(params=nnet.parameters(), lr=0.001)

    np.random.seed(1234)

    y_cali_R, y_cali_V, X_cali_R, X_cali_V = \
        train_test_split(y_cali_train, X_cali_train, test_size=0.2, random_state=1234,
                         stratify=y_cali_train)
    enc = StandardScaler().fit(X_cali_R)

    idx0_R, idx1_R = idx_I0I1(y_cali_R)

    nepochs = 100

    auc_holder = []
    for kk in range(nepochs):
        print('Epoch %i of %i' % (kk + 1, nepochs))
        # Sample class 0 pairs
        idx0_kk = np.random.choice(idx0_R, len(idx1_R), replace=False)
        for i, j in zip(idx1_R, idx0_kk):
            optimizer.zero_grad()  # clear gradient
            dlogit = nnet(torch.Tensor(enc.transform(X_cali_R[[i]]))) - \
                     nnet(torch.Tensor(
                         enc.transform(X_cali_R[[j]])))  # calculate log-odd differences
            loss = criterion(dlogit.flatten(), torch.Tensor([1]))
            loss.backward()  # backprop
            optimizer.step()  # gradient-step
        # Calculate AUC on held-out validation
        auc_k = roc_auc_score(y_cali_V,
                              nnet(torch.Tensor(
                                  enc.transform(X_cali_V))).detach().flatten().numpy())
        if auc_k > 0.9:
            print('AUC > 90% achieved')
            break

    # Compare performance on final test set
    auc_nnet_cali = roc_auc_score(y_cali_test,
                                  nnet(torch.Tensor(
                                      enc.transform(X_cali_test))).detach().flatten().numpy())

    # Fit a benchmark model
    logit_cali = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
    logit_cali.fit(enc.transform(X_cali_train), y_cali_train)
    auc_logit_cali = roc_auc_score(y_cali_test,
                                   logit_cali.predict_proba(enc.transform(X_cali_test))[:, 1])

    print('nnet-AUC: %0.3f, logit: %0.3f' % (auc_nnet_cali, auc_logit_cali))
