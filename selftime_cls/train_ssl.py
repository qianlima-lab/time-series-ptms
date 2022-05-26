# -*- coding: utf-8 -*-

import numpy as np
from evaluation.eval_ssl import evaluation
from utils.utils import get_config_from_json
import torch
import argparse
from optim.pretrain import *
import datetime
import random
from data.preprocessing import *
import os
import sys
sys.path.append('..')


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    # Bigger is better.
    parser.add_argument('--K', type=int, default=16,
                        help='Number of augmentation for each sample')

    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400,  # 400
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=400,
                        help='training patience')
    parser.add_argument('--aug_type', type=str,
                        default='none', help='Augmentation type')
    parser.add_argument('--piece_size', type=float, default=0.2,
                        help='piece size for time series piece sampling')
    parser.add_argument('--class_type', type=str,
                        default='3C', help='Classification type')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    # model dataset
    parser.add_argument('--dataset_name', type=str, default='CricketX',
                        help='dataset')
    parser.add_argument('--ucr_path', type=str, default='/dev_data/zzj/hzy/datasets/UCR',
                        help='Data root for dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    # method
    parser.add_argument('--backbone', type=str, default='SimConv4')
    parser.add_argument('--model_name', type=str, default='InterSample',
                        choices=['InterSample', 'IntraTemporal', 'SelfTime'], help='choose method')
    parser.add_argument('--config_dir', type=str,
                        default='./config', help='The Configuration Dir')
    parser.add_argument('--gpus', type=str, default='0', help='selected gpu')
    parser.add_argument('--random_seed', type=int,
                        default=42, help='for reproduction purpose')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse_option()
    exp = 'linear_eval'

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set seed
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    aug1 = ['magnitude_warp']
    aug2 = ['time_warp']

    # use uwave config
    config_dict = get_config_from_json('{}/{}_config.json'.format(
        opt.config_dir, 'UWaveGestureLibraryAll'))

    opt.class_type = config_dict['class_type']
    opt.piece_size = config_dict['piece_size']

    if opt.model_name == 'InterSample':
        model_paras = 'none'
    else:
        model_paras = '{}_{}'.format(opt.piece_size, opt.class_type)

    if aug1 == aug2:
        opt.aug_type = [aug1]
    elif type(aug1) is list:
        opt.aug_type = aug1 + aug2
    else:
        opt.aug_type = [aug1, aug2]

    log_dir = './log/{}/{}/{}/{}/{}'.format(
        exp, opt.dataset_name, opt.model_name, '_'.join(opt.aug_type), model_paras)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file2print_detail_train = open("{}/train_detail.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print_detail_train)
    print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_max\tEpoch_max",
          file=file2print_detail_train)
    file2print_detail_train.flush()

    sum_dataset, sum_target, nb_class = load_data(
        opt.ucr_path, opt.dataset_name)
    sum_dataset = np.expand_dims(sum_dataset, 2)
    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = k_fold(
        sum_dataset, sum_target)

    accu = []
    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)

    print('[INFO] Running at:', opt.dataset_name)
    save_path = './ucr_result.csv'
    for i, x_train in enumerate(train_datasets):
        print('{} fold start training!'.format(i))
        y_train = train_targets[i]
        x_val = val_datasets[i]
        y_val = val_targets[i]
        x_test = test_datasets[i]
        y_test = test_targets[i]

        x_train, x_val, x_test = fill_nan_value(x_train, x_val, x_test)
        x_train, x_val, x_test = normalize_per_series(
            x_train), normalize_per_series(x_val), normalize_per_series(x_test)
        if opt.model_name == 'InterSample':
            acc_max, epoch_max = pretrain_InterSampleRel(x_train, y_train, opt)
        elif 'IntraTemporal' in opt.model_name:
            acc_max, epoch_max = pretrain_IntraSampleRel(x_train, y_train, opt)
        elif 'SelfTime' in opt.model_name:
            acc_max, epoch_max, model_state_dict = pretrain_SelfTime(
                x_train, y_train, opt)
            acc_test, epoch_max_point = evaluation(x_train, y_train, x_val, y_val, x_test, y_test,
                                                   nb_class=nb_class, ckpt=None, opt=opt, ckpt_tosave=None, my_state=model_state_dict)

        accu.append(acc_test)

    accu = np.array(accu)
    acc_mean = np.mean(accu)
    acc_std = np.std(accu)

    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path)
    else:
        result_form = pd.DataFrame(columns=['target', 'accuracy', 'std'])

    result_form = result_form.append(
        {'target': opt.dataset_name, 'accuracy': '%.4f' % acc_mean, 'std': '%.4f' % acc_std}, ignore_index=True)
    result_form = result_form.iloc[:, -3:]
    result_form.to_csv(save_path)
