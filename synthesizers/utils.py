from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs


def benchmark(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc



def bin_DDAE(X, y):
    db = []
    df = np.hstack((X, y[:, None]))
    maj = df[y == 0]
    mio = df[y == 1]
    delta = round(len(maj) / len(mio))
    rng = np.random.default_rng()
    rand = rng.integers(low=0, high=50, size=delta)
    for i_block in range(delta):
        if i_block == delta - 1:
            db.append(np.vstack((maj, mio)))
        else:
            rng1 = np.random.default_rng(rand[i_block])
            idx = rng1.choice(len(maj), len(mio), replace=False)
            db.append(np.vstack((maj[idx], mio)))
            maj = np.delete(maj, idx, axis=0)
    # for db_ in db:
    #     np.random.shuffle(db_)
    return db


def bin_DDAE_new(X, y, delta):
    db = []
    df = np.hstack((X, y[:, None]))
    maj = df[y == 0]
    mio = df[y == 1]
    lt = int(len(df) / delta)
    rng = np.random.default_rng()
    rand = rng.integers(low=0, high=50, size=delta)
    for i_block in range(delta):
        if i_block == delta - 1:
            db.append(np.vstack((maj, mio)))
        else:
            rng1 = np.random.default_rng(rand[i_block])
            idx = rng1.choice(len(maj), lt, replace=True) # lwhay: for larger number of bins
            db.append(np.vstack((maj[idx], mio)))
            maj = np.delete(maj, idx, axis=0)
    for db_ in db:
        np.random.shuffle(db_)
    return db


def bin_predict(db: list, k: int, X, clf):
    knn = KNeighborsClassifier(n_neighbors=k)
    min_length = len(db)
    prediction_temp = []
    for block in db:
        X1 = block[:, :-1]
        y1 = block[:, -1].astype(int)
        knn.fit(clf.transform(X1), y1)
        prediction_temp.append(knn.predict(clf.transform(X)))
    prediction = np.vstack(prediction_temp)
    prediction = prediction.T.astype(int)
    maj = np.apply_along_axis(lambda x: np.bincount(x, minlength=min_length), axis=1, arr=prediction)
    y_pred = np.argmax(maj, axis=1)
    return y_pred


def bin_predict_proba(db: list, k: int, X, clf):
    knn = KNeighborsClassifier(n_neighbors=k)
    min_length = len(db)
    prediction_temp = []
    for block in db:
        X1 = block[:, :-1]
        y1 = block[:, -1]
        knn.fit(clf.transform(X1), y1)
        prediction_temp.append(knn.predict_proba(clf.transform(X)))
    proba = sum(prediction_temp) / min_length
    return proba


def norm_compute(x0, x1, gamma=1):
    # if len(x0) < len(x1):
    #     x0, x1 = x1, x0
    feature = []
    for x1_ in x1:
        norm_temp_root = np.linalg.norm(x0 - x1_, axis=1)
        norm_temp = np.square(norm_temp_root)
        feature_temp = np.exp(-gamma * norm_temp)
        feature.append(feature_temp)
    feature = np.c_[feature].T
    return np.mean(feature, axis=1)


def rbf_transform(db: list, gamma=1):
    length = len(db)
    features = []
    for i in range(length):
        features_add = []
        X = db[i][:, :-1]
        for j in range(length):
            # if i == j:
            #     continue
            # else:
            X_ = db[j][:, :-1]
            y_ = db[j][:, -1]
            X_maj_ = X_[y_ == 0]
            feature_temp = norm_compute(X, X_maj_, gamma)
            features_add.append(feature_temp)
        features_add = np.c_[features_add].T
        features.append(features_add)
    db_new = []
    for j in range(length):
        add = features[j]
        X, y = db[j][:, :-1], db[j][:, -1]
        db_new_ = np.c_[X, add, y]
        db_new.append(db_new_)
    return db_new


def rbf_transform_test(db, X_test, gamma=1):
    features = []
    for db_ in db:
        X_, y = db_[:, :-1], db_[:, -1]
        X_maj_ = X_[y == 0]
        feature_temp = norm_compute(X_test, X_maj_)
        features.append(feature_temp)
    features = np.c_[features].T
    X_test = np.hstack([X_test, features])
    return X_test


def neigh_predict(db: list, k: int, label_num: int, X, clf):
    knn = KNeighborsClassifier(n_neighbors=k)
    prediction_temp = np.zeros((len(X), label_num + 1))
    for db_ in db:
        X1, y1 = db_[:, :-1].astype(float), db_[:, -1].astype(int)
        knn.fit(clf.transform(X1), y1)
        inds = knn.kneighbors(clf.transform(X), return_distance=False)
        label = y1[inds]
        for row in range(len(label)):
            for col in range(k):
                prediction_temp[row, label[row, col]] += 1
    prediction = np.argmax(prediction_temp, axis=1)
    return prediction


def neigh_predict_remove(db: list, k: int, label_num: int, X, clf):
    knn = KNeighborsClassifier(n_neighbors=k)
    neigh = []
    prediction_temp = np.zeros((len(X), label_num + 1))
    num = 0
    for db_ in db:
        if num == 0:
            X1, y1 = db_[:, :-1].astype(float), db_[:, -1].astype(int)
            knn.fit(clf.transform(X1), y1)
            inds = knn.kneighbors(clf.transform(X), return_distance=False)
            for row in range(len(inds)):
                X_neigh = []
                for col in range(k):
                    X_neigh_ = db_[inds[row][col]]
                    X_neigh.append(X_neigh_)
                neigh.append(np.vstack((X_neigh)))
        else:
            X1, y1 = db_[:, :-1].astype(float), db_[:, -1].astype(int)
            knn.fit(clf.transform(X1), y1)
            inds = knn.kneighbors(clf.transform(X), return_distance=False)
            for row in range(len(inds)):
                X_neigh = []
                for col in range(k):
                    X_neigh_ = db_[inds[row][col]]
                    X_neigh.append(X_neigh_)
                neigh[row] = np.vstack((neigh[row], np.vstack(X_neigh)))
        num += 1
    prediction = []
    for neigh_ in neigh:
        neigh_ = np.unique(neigh_, axis=0)
        prediction_ = np.argmax(np.bincount(neigh_[:, -1].astype(int)))
        prediction.append(prediction_)
    prediction = np.array(prediction)
    return prediction


def DDAE_predict(X, transform, classifier, weight=[1, 1]):
    predictions = np.asarray([est.predict(X.dot(trans.T)) for trans, est in zip(transform, classifier)]).T
    predictions = predictions.astype(int)
    maj = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=predictions)
    maj = maj * weight
    y_pred = np.argmax(maj, axis=1)
    return y_pred

