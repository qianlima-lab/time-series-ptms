import enum
import pandas as pd
from sklearn import model_selection
import sklearn
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn import preprocessing
import numpy as np
import os
import argparse
import shutil




def load_data(dataroot, dataset):
    train = pd.read_csv(os.path.join(dataroot, dataset,
                        dataset+'_TRAIN.tsv'), sep='\t', header=None)
    train_x = train.iloc[:, 1:]
    train_target = train.iloc[:, 0]

    test = pd.read_csv(os.path.join(dataroot, dataset,
                       dataset+'_TEST.tsv'), sep='\t', header=None)
    test_x = test.iloc[:, 1:]
    test_target = test.iloc[:, 0]

    sum_dataset = pd.concat([train_x, test_x]).to_numpy(np.float32)
    #sum_dataset = sum_dataset.fillna(sum_dataset.mean()).to_np(dtype=np.float32)
    sum_target = pd.concat([train_target, test_target]).to_numpy(np.float32)
    # sum_target = sum_target.fillna(sum_target.mean()).to_np(dtype=np.float32)

    num_classes = len(np.unique(sum_target))

    #sum_target = transfer_labels(sum_target)
    return sum_dataset, sum_target


def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def k_fold(data, target):
    skf = StratifiedKFold(5, shuffle=True)
    #skf = StratifiedShuffleSplit(5)
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

        train_index, val_index = next(StratifiedKFold(
            4, shuffle=True).split(raw_set, raw_target))
        # train_index, val_index = next(StratifiedShuffleSplit(1).split(raw_set, raw_target))
        train_sets.append(raw_set[train_index])
        train_targets.append(raw_target[train_index])

        val_sets.append(raw_set[val_index])
        val_targets.append(raw_target[val_index])

    return np.array(train_sets), np.array(train_targets), np.array(val_sets), np.array(val_targets), np.array(test_sets), np.array(test_targets)


def normalize_per_series(data):
    std_ = np.std(data, axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    return (data - np.mean(data, axis=1, keepdims=True)) / std_


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

# input: dataframe after .loc[indices]
# output: a dataframe, input of the dataset_class


def fill_nan_and_normalize(train_data, val_data, test_data, train_indices, val_indices, test_indices):
    train_arr = np.array(train_data)
    train_arr = np.reshape(train_arr, [len(train_indices), -1])

    val_arr = np.array(val_data)
    val_arr = np.reshape(val_arr, [len(val_indices), -1])

    test_arr = np.array(test_data)
    test_arr = np.reshape(test_arr, [len(test_indices), -1])

    train_arr, val_arr, test_arr = fill_nan_value(train_arr, val_arr, test_arr)

    train_arr = normalize_per_series(train_arr)
    val_arr = normalize_per_series(val_arr)
    test_arr = normalize_per_series(test_arr)

    train_raw = pd.DataFrame(train_arr)
    train_df = pd.DataFrame()
    train_df['dim_0'] = [pd.Series(train_raw.iloc[x, :])
                         for x in range(len(train_raw))]
    lengths = train_df.applymap(lambda x: len(x)).values
    train_df = pd.concat((pd.DataFrame({col: train_df.loc[row, col] for col in train_df.columns}).reset_index(drop=True).set_index(
        pd.Series(lengths[row, 0]*[row])) for row in range(train_df.shape[0])), axis=0)
    train_df = train_df.groupby(train_df.index).transform(lambda x: x)

    val_raw = pd.DataFrame(val_arr)
    val_df = pd.DataFrame()
    val_df['dim_0'] = [pd.Series(val_raw.iloc[x, :])
                       for x in range(len(val_raw))]
    lengths = val_df.applymap(lambda x: len(x)).values
    val_df = pd.concat((pd.DataFrame({col: val_df.loc[row, col] for col in val_df.columns}).reset_index(drop=True).set_index(
        pd.Series(lengths[row, 0]*[row])) for row in range(val_df.shape[0])), axis=0)
    val_df = val_df.groupby(val_df.index).transform(lambda x: x)

    test_raw = pd.DataFrame(test_arr)
    test_df = pd.DataFrame()
    test_df['dim_0'] = [pd.Series(test_raw.iloc[x, :])
                        for x in range(len(test_raw))]
    lengths = test_df.applymap(lambda x: len(x)).values
    test_df = pd.concat((pd.DataFrame({col: test_df.loc[row, col] for col in test_df.columns}).reset_index(drop=True).set_index(
        pd.Series(lengths[row, 0]*[row])) for row in range(test_df.shape[0])), axis=0)
    test_df = test_df.groupby(test_df.index).transform(lambda x: x)

    return train_df, val_df, test_df


if __name__ == '__main__':
    '''
    CACHE_PATH = './src/data_cache'
    shutil.rmtree(CACHE_PATH)
    os.mkdir(CACHE_PATH)

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', default='/dev_data/zzj/hzy/datasets/UCR', type=str)
    parser.add_argument('--dataset', default='ArrowHead', type=str)

    args = parser.parse_args()

    sum_dataset, sum_target = load_data(args.dataroot, args.dataset)
    print(sum_target)
    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = k_fold(sum_dataset, sum_target)

    for i, train_dataset in enumerate(train_datasets):
        train_target = train_targets[i]

        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]


    test1 = pd.DataFrame(train_datasets[0])
    test2 = pd.DataFrame(train_targets[0])

    out = pd.concat([test2, test1], axis=1, ignore_index=True)
    out.to_csv(os.path.join(CACHE_PATH, 'temp.tsv'), sep='\t', index=False, header=False)
    ds, target = sktime.utils.load_data.load_from_ucr_tsv_to_dataframe(os.path.join(CACHE_PATH, 'temp.tsv'), return_separate_X_and_y=True)

    print(ds)
    print(target)
    '''
    sklearn.random.seed(42)
    sum_dataset, sum_target = load_data(
        '/dev_data/zzj/hzy/datasets/UCR', 'ArrowHead')
    skf = model_selection.StratifiedKFold(5, shuffle=True, random_state=42)

    for x, y in skf.split(sum_dataset, sum_target):
        print(x, y)
