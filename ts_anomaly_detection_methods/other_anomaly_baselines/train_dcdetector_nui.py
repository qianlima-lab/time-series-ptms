import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from other_anomaly_baselines.dcdetector_solver import Solver
import time
import warnings
import sys
from other_anomaly_baselines.datasets.data_loader import get_loader_segment

import datautils
from torch.utils.data import TensorDataset, DataLoader


import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore')


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



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def str2bool(v):
    return v.lower() in ('true')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(array[idx - 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Alternative
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--patch_size', type=list, default=[5])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss_fuc', type=str, default='MSE')
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--rec_timeseries', action='store_true', default=True)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # Default
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_c', type=int, default=1)
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--k', type=int, default=3)
    # parser.add_argument('--dataset', type=str, default='NIPS_TS_Swan') ## NIPS_TS_Swan  SMD
    parser.add_argument('--dataset', type=str, default='yahoo')  ##  kpi, yahoo
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='datasets/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    parser.add_argument('--anormly_ratio', type=float, default=1.00)

    parser.add_argument('--save_dir', type=str, default='/dev_data/lz/tsm_ptms_anomaly_detection/result/')
    parser.add_argument('--save_csv_name', type=str, default='dcdetector_uni_0722.csv')

    config = parser.parse_args()
    args = vars(config)
    config.patch_size = [int(patch_index) for patch_index in config.patch_size]

    # 检查路径是否存在，如果不存在则赋值为新的路径
    if not os.path.exists(config.save_dir):
        config.save_dir = '/SSD/lz/tsm_ptms_anomaly_detection/result/'

    print("save_dir = ", config.save_dir)  # 输出检查

    # if config.dataset == 'UCR':
    #     batch_size_buffer = [2, 4, 8, 16, 32, 64, 128, 256]
    #     data_len = np.load(config.data_path + config.dataset + "/UCR_" + str(config.index) + "_train.npy").shape[0]   ## './datasets/' +
    #     config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)
    # elif config.dataset == 'UCR_AUG':
    #     batch_size_buffer = [2, 4, 8, 16, 32, 64, 128, 256]
    #     data_len = np.load('./datasets/' + config.data_path + "/UCR_AUG_" + str(config.index) + "_train.npy").shape[0]
    #     config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)
    # elif config.dataset == 'SMD_Ori':
    #     batch_size_buffer = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    #     data_len = np.load('./datasets/' + config.data_path + "/SMD_Ori_" + str(config.index) + "_train.npy").shape[0]
    #     config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)

    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]

    sys.stdout = Logger("./result_log/" + config.dataset + ".log", sys.stdout)
    if config.mode == 'train':
        print("\n\n")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('================ Hyperparameters ===============')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('====================  Train  ===================')

    # train_loader, train_set = get_loader_segment(config.index, config.data_path + config.dataset, batch_size=config.batch_size,
    #                                              win_size=config.win_size, mode='train', dataset=config.dataset)
    #
    # train_set = train_set.train
    #
    #
    # print("train_set.shape = ", train_set.shape)
    # config.input_c = train_set.shape[-1]
    # config.output_c = train_set.shape[-1]

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

    val_loader = train_loader

    config.input_c = train_data.shape[-1]
    config.output_c = train_data.shape[-1]

    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    solver.train_loader = train_loader
    solver.vali_loader = val_loader
    solver.test_loader = val_loader
    solver.thre_loader = val_loader

    solver.train_uni()
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


