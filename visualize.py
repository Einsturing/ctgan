import matplotlib.pyplot as plt
import numpy as np


def plot_2d(X, y, Xt, Yt, label='distribution', xlimits=[-7, 17], ylimits=[-7, 17]):
    plt.subplots(figsize=(24, 18))
    print('distribution')
    print(X.shape)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(label)
    plt.legend(loc='upper right')
    plt.xlim(xlimits[0], xlimits[1])
    plt.ylim(ylimits[0], ylimits[1])
    plt.savefig('dist.png')
    plt.show()


def plot_2d_TF(X, yt, yp, s=5, label='Classes', xlimits=[-7, 17], ylimits=[-7, 17]):
    plt.subplots(figsize=(24, 18))
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 'o']
    for l, c, m in zip(np.unique(yt), colors, markers):
        plt.scatter(
            X[yt == l, 0],
            X[yt == l, 1],
            c=c, label=l, marker=m, s=s
        )
    ym = yt - yp
    yT = X[ym == -1]
    yF = X[ym == 1]
    plt.scatter(
        yT[:, 0],
        yT[:, 1],
        c='#FF0000', label=l, marker='o', s=s * 2
    )
    plt.scatter(
        yF[:, 0],
        yF[:, 1],
        c='#0000FF', label=l, marker='o', s=s * 2
    )
    plt.title(label, fontdict={'fontsize': 48})
    plt.legend(loc='upper right')
    plt.xlim(xlimits[0], xlimits[1])
    plt.ylim(ylimits[0], ylimits[1])
    plt.savefig('output/' + label + '-tf.png')
    plt.show()


def plot_2d_space(X, y, Xt, yt, dif=True, s=5, label='Classes', xlimits=[-7, 17], ylimits=[-7, 17]):
    # print(X, y)
    # pca = PCA(n_components=3)
    # X = pca.fit_transform(X)
    plt.subplots(figsize=(24, 18))
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 'o']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m, s=s
        )

    # colors = ['#00FF7F', '#FFFF0E']
    # if dif:
    #     markers = ['x', '+']
    # for l, c, m in zip(np.unique(y), colors, markers):
    #     plt.scatter(
    #         Xt[yt == l, 0],
    #         Xt[yt == l, 1],
    #         c=c, label=l, marker=m, s=s
    #     )
    plt.title(label, fontdict={'fontsize': 48})
    plt.legend(loc='upper right')
    plt.xlim(xlimits[0], xlimits[1])
    plt.ylim(ylimits[0], ylimits[1])
    # plt.savefig('output/' + label + '-2d.png')
    plt.show()


def plot_3d_space(X, y, Xt=[], yt=[], label='Classes'):
    # pca = PCA(n_components=8)
    # X = pca.fit_transform(X)
    ax = plt.figure().add_subplot(111, projection='3d')
    colors = ['#00007F', '#FF7F0E']
    markers = ['o', 'o']
    for l, c, m in zip(np.unique(y), colors, markers):
        ax.scatter(
            X[y == l, 0],
            X[y == l, 1],
            X[y == l, 2],
            c=c, label=l, marker=m
        )
    colors = ['#00007F', '#FF7F0E']
    markers = ['+', '+']
    for l, c, m in zip(np.unique(yt), colors, markers):
        ax.scatter(
            Xt[yt == l, 0],
            Xt[yt == l, 1],
            Xt[yt == l, 2],
            c=c, label=l, marker=m
        )
    plt.title(label, fontdict={'fontsize': 18})
    plt.legend(loc='upper right')
    plt.show()


def plot_3d_space3(X,y,Xt=[],yt=[],label='classes'):
    ax = plt.figure().add_subplot(111, projection='3d')
    colors = ['#00007F', '#FF7F0E']
    markers = ['o', 'o']
    for l, c, m in zip(np.unique(y), colors, markers):
        ax.scatter(
            X[y == l, 0],
            X[y == l, 1],
            X[y == l, 2],
            c=c, label=l, marker=m
        )
    colors = ['#00007F', '#FF7F0E']
    markers = ['+', '+']
    for l, c, m in zip(np.unique(yt), colors, markers):
        ax.scatter(
            Xt[yt == l, 0],
            Xt[yt == l, 1],
            Xt[yt == l, 2],
            c=c, label=l, marker=m
        )
    plt.title(label, fontdict={'fontsize': 18})
    plt.legend(loc='upper right')
    plt.show()
