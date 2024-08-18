import os
import sys

import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import os
import argparse
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.backends import cudnn
from other_anomaly_baselines.datasets.data_loader import get_loader_segment
from other_anomaly_baselines.AT_solver import Solver, mkdir
import datautils
import numpy as np
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')



class UniLoader(object):
    def __init__(self, data_set, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size

        self.train = data_set


    def __len__(self):
        """
        Number of images in the object dataset.
        """

        return (self.train.shape[0] - self.win_size) // self.step + 1


    def __getitem__(self, index):
        index = index * self.step

        return np.float32(self.train[index:index + self.win_size])


def str2bool(v):
    return v.lower() in ('true')


def main(config, train_set, train_loader, val_set, val_loader, test_set, test_loader, dev_cuda, all_train_data, all_test_data, all_test_labels, all_test_timestamps, delay, train_data):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)

    for i in range(train_data.shape[0]):
        print("i = ", i, ", total num = ", train_data.shape[0])
        print("train_data.shape = ", train_data.shape)
        _train_data = train_data[i]
        print("000train_data.shape = ", train_data.shape, type(train_data))
        _train_data = np.array(_train_data)
        print("111_train_data.shape = ", _train_data.shape, type(_train_data))

        train_dataset = UniLoader(_train_data, config.win_size, 1)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)

        solver = Solver(vars(config), train_dataset, train_loader, val_set, val_loader, test_set, test_loader, dev_cuda)

        break

    # if config.mode == 'train':
    # for _uni_train_set in train_set:

    # solver.train_uni()
    # elif config.mode == 'test':
    eval_res = solver.test_uni(all_train_data, all_test_data, all_test_labels, all_test_timestamps, delay, config)

    print("result_dict = ", eval_res)

    eval_res['dataset'] = config.dataset + str(config.index)
    import pandas as pd

    # 转换字典为 DataFrame
    df = pd.DataFrame([eval_res])
    # 指定保存路径
    save_path = config.save_dir + config.save_csv_name

    # 转换字典为 DataFrame
    df_new = pd.DataFrame([eval_res])

    # 检查文件是否存在
    if os.path.exists(save_path):
        # 文件存在，读取现有数据
        df_existing = pd.read_csv(save_path, index_col=0)
        # 将新数据附加到现有数据框中
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # 文件不存在，创建新的数据框
        df_combined = df_new

    # 保存 DataFrame 为 CSV 文件
    df_combined.to_csv(save_path, index=True, index_label="id")

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='yahoo')    ##  kpi, yahoo
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--data_path', type=str, default='datasets/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=1.0)
    parser.add_argument('--index', type=int, default=143, help='')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    parser.add_argument('--save_dir', type=str, default='/dev_data/lz/tsm_ptms_anomaly_detection/result/')
    parser.add_argument('--save_csv_name', type=str, default='at_uni_0722.csv')

    config = parser.parse_args()

    # 检查路径是否存在，如果不存在则赋值为新的路径
    if not os.path.exists(config.save_dir):
        config.save_dir = '/SSD/lz/tsm_ptms_anomaly_detection/result/'

    print("save_dir = ", config.save_dir)  # 输出检查

    dataset = 'MSL'
    _train_loader, _train_set = get_loader_segment(config.index, config.data_path + dataset,
                                                 batch_size=config.batch_size,
                                                 win_size=config.win_size, mode='train', dataset=dataset)

    _train_set = _train_set.train

    print("_train_set.shape = ", _train_set.shape)

    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(
        config.dataset)
    train_data = datautils.gen_ano_train_data(all_train_data)

    print("train_data.shape = ", train_data.shape)
    _train_data = train_data[0]
    print("000train_data.shape = ", train_data.shape, type(train_data))
    _train_data = np.array(_train_data)
    print("111_train_data.shape = ", _train_data.shape, type(_train_data))

    train_dataset = UniLoader(_train_data, config.win_size, 1)

    train_loader = DataLoader(dataset=train_dataset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=2,
                             drop_last=True)

    # train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
    #                           drop_last=True)

    val_loader = train_loader

    config.input_c = train_data.shape[-1]
    config.output_c = train_data.shape[-1]

    args = vars(config)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config, train_dataset, train_loader, train_dataset, val_loader, train_dataset, val_loader, config.cuda, all_train_data, all_test_data,
         all_test_labels, all_test_timestamps, delay, train_data)
