import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def load_data(dataroot, dataset):
    train = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TRAIN.tsv'), sep='\t', header=None)
    train_x = train.iloc[:, 1:]
    train_target = train.iloc[:, 0]

    test = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TEST.tsv'), sep='\t', header=None)
    test_x = test.iloc[:, 1:]
    test_target = test.iloc[:, 0]

    sum_dataset = pd.concat([train_x, test_x]).to_numpy(dtype=np.float32)
    # sum_dataset = sum_dataset.fillna(sum_dataset.mean()).to_numpy(dtype=np.float32)
    sum_target = pd.concat([train_target, test_target]).to_numpy(dtype=np.float32)
    # sum_target = sum_target.fillna(sum_target.mean()).to_numpy(dtype=np.float32)

    num_classes = len(np.unique(sum_target))

    return sum_dataset, sum_target, num_classes


def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def k_fold(data, target):
    skf = StratifiedKFold(5, shuffle=True)
    # skf = StratifiedShuffleSplit(5)
    train_sets = []
    train_targets = []

    val_sets = []
    val_targets = []

    test_sets = []
    test_targets = []

    for raw_index, test_index in skf.split(data, target):
        raw_set = data[raw_index]
        raw_target = target[raw_index]

        test_sets.append(data[test_index])
        test_targets.append(target[test_index])

        train_index, val_index = next(StratifiedKFold(4, shuffle=True).split(raw_set, raw_target))
        # train_index, val_index = next(StratifiedShuffleSplit(1).split(raw_set, raw_target))
        train_sets.append(raw_set[train_index])
        train_targets.append(raw_target[train_index])

        val_sets.append(raw_set[val_index])
        val_targets.append(raw_target[val_index])

    return train_sets, train_targets, val_sets, val_targets, test_sets, test_targets


def normalize_per_series(data):
    std_ = data.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    return (data - data.mean(axis=1, keepdims=True)) / std_


def normalize_train_val_test(train_set, val_set, test_set):
    mean = train_set.mean()
    std = train_set.std()
    return (train_set - mean) / std, (val_set - mean) / std, (test_set - mean) / std


def fill_nan_value(train_set, val_set, test_set):
    ind = np.where(np.isnan(train_set))
    col_mean = np.nanmean(train_set, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_set[ind] = np.take(col_mean, ind[1])

    ind_val = np.where(np.isnan(val_set))
    val_set[ind_val] = np.take(col_mean, ind_val[1])

    ind_test = np.where(np.isnan(test_set))
    test_set[ind_test] = np.take(col_mean, ind_test[1])
    return train_set, val_set, test_set


if __name__ == '__main__':
    pass
