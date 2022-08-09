import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.tree import DecisionTreeClassifier
import warnings
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from .utils import benchmark

warnings.filterwarnings("ignore")


class LRSample:
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 hardness_func=lambda y_true, y_pred: np.absolute(
                     y_true - y_pred),
                 n_estimators=10,
                 k_bins=10,
                 random_state=None):
        self.X_sample_min = []
        self.base_estimator_ = base_estimator
        self.estimators_ = []
        self._hardness_func = hardness_func
        self._n_estimators = n_estimators
        self._k_bins = k_bins
        self._random_state = random_state
        self.max_auc = 0

    def _fit_base_estimator(self, X, y):
        return sklearn.base.clone(self.base_estimator_).fit(X, y)

    def _random_under_sampling(self, X_maj, y_maj, X_min, y_min):
        np.random.seed(self._random_state)
        idx = np.random.choice(len(X_maj), len(X_min), replace=True)
        X_train = np.concatenate([X_maj[idx], X_min])
        # X_train = torch.cat((X_maj[idx], X_min), dim=0)
        y_train = np.concatenate([y_maj[idx], y_min])
        return X_train, y_train, idx

    def _self_paced_under_sampling_P(self, X_sample, X_cls, y_sample, X_tar, y_tar, i_estimator, y_pred, maj_bins):
        hardness = self._hardness_func(y_sample, y_pred)
        if hardness.max() == hardness.min():
            # X_train, y_train = self._random_under_sampling(X_sample, y_sample, X_tar, y_tar)
            return None, None, None, None, None
        else:
            step = (hardness.max() - hardness.min()) / self._k_bins
            bins = []
            bins_cls = []
            bins_index = []
            ave_contributions = []
            X_index = np.expand_dims(np.linspace(0, len(X_cls) - 1, num=len(X_cls), dtype=np.int), axis=1)
            for i_bins in range(self._k_bins):
                idx = (
                        (hardness >= i_bins * step + hardness.min()) &
                        (hardness < (i_bins + 1) * step + hardness.min())
                )
                if i_bins == (self._k_bins - 1):
                    idx = idx | (hardness == hardness.max())
                bins.append(X_sample[idx])
                bins_cls.append(X_cls[idx])
                bins_index.append(X_index[idx])
                ave_contributions.append(hardness[idx].mean())
            alpha = np.tan(np.pi * 0.5 * (i_estimator / (self._n_estimators - 1)))
            weights = 1 / (ave_contributions + alpha)
            weights[np.isnan(weights)] = 0
            n_sample_bins = len(X_tar) * weights / weights.sum() / 2
            n_sample_bins = n_sample_bins.astype(int) + 1
            # maj_bins_sum = 0
            # for i in range(10):
            #     maj_bins_sum += len(maj_bins[i])
            # for i in range(10):
            #     n_sample_bins[i] = len(maj_bins[i]) * len(X_tar) / maj_bins_sum / 10
            sampled_bins = []
            sampled_bins_cls = []
            sampled_bins_index = []
            # for bin in bins:
            #     print(len(bin))
            # print('\n')
            for i_bins in range(self._k_bins):
                if min(len(bins[i_bins]), n_sample_bins[i_bins]) > 0:
                    np.random.seed(self._random_state)
                    idx = np.random.choice(
                        len(bins[i_bins]),
                        min(len(bins[i_bins]), n_sample_bins[i_bins]),  # TODO:not 1:1
                        replace=False)
                    sampled_bins.append(bins[i_bins][idx])
                    sampled_bins_cls.append(bins_cls[i_bins][idx])
                    sampled_bins_index.append(bins_index[i_bins][idx])
            # for sbin in sampled_bins:
            #     print(len(sbin))
            # print('\n')
            X_train_cls = np.concatenate(sampled_bins_cls, axis=0)
            X_train = torch.cat(sampled_bins, dim=0)
            X_train_index = np.concatenate(sampled_bins_index, axis=0)
        return X_train, np.ones(len(X_train)), X_train_cls, X_train_index, bins
        # return sampled_bins[:], np.ones(len(sampled_bins[0])), sampled_bins_cls[0]

    def _self_paced_under_sampling_N(self, X_sample, y_sample, X_tar, y_tar, i_estimator, y_pred, min_bins):
        hardness = self._hardness_func(y_sample, y_pred)
        if hardness.max() == hardness.min():
            # X_train, y_train = self._random_under_sampling(X_sample, y_sample, X_tar, y_tar)
            return None, None, None, None
        else:
            step = (hardness.max() - hardness.min()) / self._k_bins
            bins = []
            bins_index = []
            X_index = np.expand_dims(np.linspace(0, len(X_sample) - 1, num=len(X_sample), dtype=np.int), axis=1)
            ave_contributions = []
            for i_bins in range(self._k_bins):
                idx = (
                        (hardness >= i_bins * step + hardness.min()) &
                        (hardness < (i_bins + 1) * step + hardness.min())
                )
                if i_bins == (self._k_bins - 1):
                    idx = idx | (hardness == hardness.max())
                bins.append(X_sample[idx])
                bins_index.append(X_index[idx])
                ave_contributions.append(hardness[idx].mean())
            alpha = np.tan(np.pi * 0.5 * (i_estimator / (self._n_estimators - 1)))
            weights = 1 / (ave_contributions + alpha)
            weights[np.isnan(weights)] = 0
            n_sample_bins = len(X_tar) * weights / weights.sum() / 2
            n_sample_bins = n_sample_bins.astype(int) + 1
            sampled_bins = []
            sampled_bins_index = []
            # min_bins_sum = 0
            # for i in range(10):
            #     min_bins_sum += len(min_bins[i])
            # for i in range(10):
            #     n_sample_bins[i] = len(min_bins[i]) * len(X_tar) / min_bins_sum / 10
            # for bin in bins:
            #     print(len(bin))
            # print('\n')
            for i_bins in range(self._k_bins):
                if min(len(bins[i_bins]), n_sample_bins[i_bins]) > 0:
                    np.random.seed(self._random_state)
                    idx = np.random.choice(
                        len(bins[i_bins]),
                        min(len(bins[i_bins]), n_sample_bins[i_bins]),  # TODO:not 1:1
                        replace=False)
                    sampled_bins.append(bins[i_bins][idx])
                    sampled_bins_index.append(bins_index[i_bins][idx])
            X_train = np.concatenate(sampled_bins, axis=0)
            X_train_index = np.concatenate(sampled_bins_index, axis=0)
            # for sbin in sampled_bins:
            #     print(len(sbin))
            # print('\n')
        return X_train, np.zeros(len(X_train)), X_train_index, bins
        # return bins[-1], np.zeros(len(bins[-1])), bins

    def _init_bins(self, X_sample, y_sample, y_pred):
        hardness = self._hardness_func(y_sample, y_pred)
        if hardness.max() == hardness.min():
            return None, None
        else:
            step = (hardness.max() - hardness.min()) / self._k_bins
            bins = []
            for i_bins in range(self._k_bins):
                idx = (
                        (hardness >= i_bins * step + hardness.min()) &
                        (hardness < (i_bins + 1) * step + hardness.min())
                )
                if i_bins == (self._k_bins - 1):
                    idx = idx | (hardness == hardness.max())
                bins.append(X_sample[idx])
        return bins

    def del_tensor_ele(self, arr, index):
        res = []
        index = np.unique(index)
        res.append(arr[:index[0]])
        for i in range(1, len(index)):
            precut = index[i - 1]
            cut = index[i]
            res.append(arr[precut + 1:cut])
        res.append(arr[index[-1] + 1:])
        return torch.cat(res, dim=0)

    def fit(self, data_ori, data_gan, data_sample, data_test, label_maj=0, label_min=1):
        self.estimators_ = []
        X_ori = data_ori[:, :2]
        y_ori = data_ori[:, -1]
        X_gan = data_gan[:, :2]
        y_gan = data_gan[:, -1]
        X_ori_maj = X_ori[y_ori == label_maj]
        y_ori_maj = y_ori[y_ori == label_maj]
        X_ori_min = X_ori[y_ori == label_min]
        y_ori_min = y_ori[y_ori == label_min]
        X_gan_min = X_gan[y_gan == label_min]
        y_gan_min = y_gan[y_gan == label_min]
        X_sample_maj = data_sample[y_gan == label_maj]
        X_sample_min = data_sample[y_gan == label_min]
        X_test = data_test[:, :2]
        y_test = data_test[:, -1]
        X_D0, y_D0, idx = self._random_under_sampling(
            X_ori_maj, y_ori_maj, X_ori_min, y_ori_min)
        X_ori_maj = np.delete(X_ori_maj, idx, axis=0)
        y_ori_maj = np.delete(y_ori_maj, idx, axis=0)
        self.estimators_.append(
            self._fit_base_estimator(
                X_D0, y_D0))
        self._y_pred_gan = self.estimators_[-1].predict_proba(X_gan_min)[:, 1]
        self._y_pred_maj = self.estimators_[-1].predict_proba(X_ori_maj)[:, 1]
        maj_bins = self._init_bins(
            X_ori_maj, y_ori_maj, self._y_pred_maj)
        # self._y_pred_maj = self.predict_proba(X_ori_maj)[:, 1]
        tmp_sample_min = []
        tmp_sample_min_cls = []
        for i_estimator in range(1, self._n_estimators):
            if len(X_ori_maj) == 0:
                break
            # print(i_estimator)
            # vec = []
            # for i in range(len(maj_bins)):
            #     vec.append(len(maj_bins[i]))
            # print(vec)
            X_Pi, y_Pi, X_Pi_cls, X_Pi_index, min_bins = self._self_paced_under_sampling_P(
                X_sample_min, X_gan_min, y_gan_min, X_ori_min, y_ori_min, i_estimator, self._y_pred_gan, maj_bins)
            # vec = []
            # for i in range(len(min_bins)):
            #     vec.append(len(min_bins[i]))
            # print(vec)
            # plt.scatter(X_Pi[:, 0], X_Pi[:, 1], s=10)
            # plt.xlim(-4, 14)
            # plt.ylim(-8, 16)
            # plt.show()
            if X_Pi is None:
                break
            X_sample_min = self.del_tensor_ele(X_sample_min, X_Pi_index)
            X_gan_min = np.delete(X_gan_min, X_Pi_index, axis=0)
            y_gan_min = np.delete(y_gan_min, X_Pi_index, axis=0)
            X_D0 = np.append(X_D0, X_Pi_cls, axis=0)
            y_D0 = np.append(y_D0, y_Pi, axis=0)

            tmp_sample_min.append(X_Pi)
            tmp_sample_min_cls.append(X_Pi_cls)
            # self.X_train = torch.cat((self.X_train, X_train), dim=0)
            self.estimators_.append(
                self._fit_base_estimator(
                    X_D0, y_D0))
            # n_clf = len(self.estimators_)
            # y_pred_gan_last_clf = self.estimators_[-1].predict_proba(X_gan_min)[:, 1]
            # self._y_pred_gan = (self._y_pred_gan * (n_clf - 1) + y_pred_gan_last_clf) / n_clf
            self._y_pred_maj = self.estimators_[-1].predict_proba(X_ori_maj)[:, 1]
            X_Ni, y_Ni, X_Ni_index, maj_bins = self._self_paced_under_sampling_N(
                X_ori_maj, y_ori_maj, X_ori_min, y_ori_min, i_estimator, self._y_pred_maj, min_bins)
            if X_Ni is None:
                break
            X_ori_maj = np.delete(X_ori_maj, X_Ni_index, axis=0)
            y_ori_maj = np.delete(y_ori_maj, X_Ni_index, axis=0)
            X_D0 = np.append(X_D0, X_Ni, axis=0)
            y_D0 = np.append(y_D0, y_Ni, axis=0)
            self.estimators_.append(
                self._fit_base_estimator(
                    X_D0, y_D0))
            # n_clf = len(self.estimators_)
            # y_pred_maj_last_clf = self.estimators_[-1].predict_proba(X_ori_maj)[:, 1]
            # self._y_pred_maj = (self._y_pred_maj * (n_clf - 1) + y_pred_maj_last_clf) / n_clf
            self._y_pred_gan = self.estimators_[-1].predict_proba(X_gan_min)[:, 1]

            test_sample_min = np.concatenate(tmp_sample_min_cls, axis=0)
            y_sample_min = np.ones(len(test_sample_min))
            X_cat = np.vstack([X_ori, test_sample_min])
            y_cat = np.hstack([y_ori, y_sample_min])
            clf = GradientBoostingClassifier()
            clf.fit(X_cat, y_cat)
            y_pred = clf.predict(X_test)
            tmp_auc = benchmark(y_test, y_pred)
            if tmp_auc > self.max_auc:
                self.max_auc = tmp_auc
                self.X_sample_min = tmp_sample_min.copy()
        self.X_sample_min = torch.cat(self.X_sample_min, dim=0)
        return self

    def predict_proba(self, X):
        y_pred = np.array(
            [model.predict(X) for model in self.estimators_]
        ).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)
        return y_pred

    def predict(self, X):
        y_pred_binarized = sklearn.preprocessing.binarize(
            self.predict_proba(X)[:, 1].reshape(1, -1), threshold=0.2)[0]
        return y_pred_binarized

    def score(self, X, y):
        return sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, 1])

    def get_params(self, deep=False):
        return {'base_estimator': self.base_estimator_}
