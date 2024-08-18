import os
import torch.utils.data as data
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import random
import torch
import torch.nn as nn


def build_dataset(args):
    sum_dataset, sum_target, num_classes = load_data(args.dataroot, args.dataset)

    sum_target = transfer_labels(sum_target)
    return sum_dataset, sum_target, num_classes


def load_data(dataroot, dataset):
    train = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TRAIN.tsv'), sep='\t', header=None)
    train_x = train.iloc[:, 1:]
    train_target = train.iloc[:, 0]

    test = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TEST.tsv'), sep='\t', header=None)
    test_x = test.iloc[:, 1:]
    test_target = test.iloc[:, 0]

    sum_dataset = pd.concat([train_x, test_x]).to_numpy(dtype=np.float32)
    sum_target = pd.concat([train_target, test_target]).to_numpy(dtype=np.float32)

    num_classes = len(np.unique(sum_target))

    return sum_dataset, sum_target, num_classes


def normalize_per_series(data):
    std_ = data.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    return (data - data.mean(axis=1, keepdims=True)) / std_



def load_UEA(dataroot, dataset):
    '''
    scipy 1.3.0 or newer is required to load. Otherwise, the data cannot be loaded.
    '''
    train_data = loadarff(os.path.join(dataroot, dataset, dataset + '_TRAIN.arff'))[0]
    test_data = loadarff(os.path.join(dataroot, dataset, dataset + '_TEST.arff'))[0]

    def extract_data(data_set):
        res_data = []
        res_labels = []
        for t_data, t_label in data_set:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    sum_dataset = np.concatenate((train_X, test_X), axis=0,
                                 dtype=np.float32)  # (num_size, series_length, num_dimensions)
    sum_target = np.concatenate((train_y, test_y), axis=0, dtype=np.float32)
    num_classes = len(np.unique(sum_target))
    return sum_dataset, sum_target, num_classes


def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def k_fold(data_set, target):
    skf = StratifiedKFold(5, shuffle=True)
    # skf = StratifiedShuffleSplit(5)
    train_sets = []
    train_targets = []

    val_sets = []
    val_targets = []

    test_sets = []
    test_targets = []

    for raw_index, test_index in skf.split(data_set, target):
        raw_set = data_set[raw_index]
        raw_target = target[raw_index]

        test_sets.append(data_set[test_index])
        test_targets.append(target[test_index])

        train_index, val_index = next(StratifiedKFold(4, shuffle=True).split(raw_set, raw_target))
        # train_index, val_index = next(StratifiedShuffleSplit(1).split(raw_set, raw_target))
        train_sets.append(raw_set[train_index])
        train_targets.append(raw_target[train_index])

        val_sets.append(raw_set[val_index])
        val_targets.append(raw_target[val_index])

    return train_sets, train_targets, val_sets, val_targets, test_sets, test_targets


def normalize_uea_set(data_set):
    '''
    The function is the same as normalize_per_series, but can be used for multiple variables.
    '''
    return TimeSeriesScalerMeanVariance().fit_transform(data_set)


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


class UEADataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset.permute(0, 2, 1)  # (num_size, num_dimensions, series_length)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)


def save_cls_new_result(args, mean_accu, max_acc, min_acc, std_acc, train_time):
    save_path = os.path.join(args.save_dir, '', args.save_csv_name + '_sup_cls_result.csv')
    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path, index_col=0)
    else:
        result_form = pd.DataFrame(
            columns=['dataset_name', 'mean_accu', 'max_acc', 'min_acc', 'std_acc', 'train_time'])

    result_form = result_form.append(
        {'dataset_name': args.dataset, 'mean_accu': '%.4f' % mean_accu, 'max_acc': '%.4f' % max_acc,
         'min_acc': '%.4f' % min_acc,
         'std_acc': '%.4f' % std_acc,
         'train_time': '%.4f' % train_time
         }, ignore_index=True)

    result_form.to_csv(save_path, index=True, index_label="id")


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def get_all_datasets(data_set, target):
    return k_fold(data_set, target)



def cross_entropy():
    loss = nn.CrossEntropyLoss()
    return loss


def reconstruction_loss():
    loss = nn.MSELoss()
    return loss


def build_loss(args):
    if args.loss == 'cross_entropy':
        return cross_entropy()
    elif args.loss == 'reconstruction':
        return reconstruction_loss()