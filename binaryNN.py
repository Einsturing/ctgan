import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import benchmark

import matplotlib.pyplot as plt

N = 1000
df = pd.read_csv('gen.csv')
width = df.shape[1] - 1
print(width)
# X, y = df.values[:1000, :-1].astype(float), df.values[:1000, -1].astype(int)
X, y = df.values[:, :-1].astype(float), df.values[:, -1].astype(int)
active = np.reshape(np.array([1, 0] * len(X)), (len(X), -1))
X = np.concatenate((X, active), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, stratify=y, random_state=42)
# plot_2d_space(X_train, y_train, X_test, y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = y_train
X, Y = X_train, y_train
d = []
for y in Y:
    d.append([y])

X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(width + 2, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


net = Net()
opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.BCELoss()


def train_epoch(model, opt, criterion, batch_size=50):
    model.train()
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = Y[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = net(x_batch)
        # (2) Compute diff
        loss = criterion(y_hat.squeeze(), y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())
    return losses


e_losses = []
num_epochs = 200
for e in range(num_epochs):
    e_losses += train_epoch(net, opt, criterion)
plt.plot(e_losses)

net.eval()

df1 = pd.read_csv('real.csv')
X, y = df1.values[:1000, :-1].astype(float), df1.values[:1000, -1].astype(int)
active = np.reshape(np.array([1, 0] * len(X)), (len(X), -1))
X = np.concatenate((X, active), axis=1)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.99, stratify=y)
X_test = scaler.transform(X_test1)
y_test = y_test1
x_t = torch.from_numpy(X_test).float()
y_pred = net(x_t)
out = y_pred.detach().numpy()
for i in range(np.size(out)):
    print(out[i][0], y_test[i])
y_pred1 = (y_pred.detach().numpy() > 0.5).astype(int).flatten()
benchmark(y_test, y_pred1)
print(roc_auc_score(y_test, y_pred.detach().numpy()))
# x_t = Variable(torch.randn(50, 31))
# print(net(x_t))
# x_1_t = Variable(torch.randn(50, 31) + 1.5)
# print(net(x_1_t))
