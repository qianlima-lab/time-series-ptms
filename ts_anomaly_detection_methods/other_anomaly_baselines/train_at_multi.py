import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import os
import argparse

from torch.backends import cudnn
from other_anomaly_baselines.datasets.data_loader import get_loader_segment
from other_anomaly_baselines.AT_solver import Solver, mkdir
import torch.multiprocessing as mp
import numpy as np

# 更改共享策略
mp.set_sharing_strategy('file_system')



def str2bool(v):
    return v.lower() in ('true')


def main(config, train_set, train_loader, val_set, val_loader, test_set, test_loader, dev_cuda):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config), train_set, train_loader, val_set, val_loader, test_set, test_loader, dev_cuda)

    # if config.mode == 'train':
    solver.train()
    # elif config.mode == 'test':
    eval_res = solver.test(ucr_index=config.index)

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
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='UCR')    ##  SMD, MSL, SMAP, PSM, SWAT, NIPS_TS_Swan, UCR, NIPS_TS_Water, UCR
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--data_path', type=str, default='datasets/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=0.9)
    parser.add_argument('--index', type=int, default=143, help='')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    parser.add_argument('--save_dir', type=str, default='/dev_data/lz/tsm_ptms_anomaly_detection/result/')
    parser.add_argument('--save_csv_name', type=str, default='at_ucr_0727.csv')

    config = parser.parse_args()

    # 检查路径是否存在，如果不存在则赋值为新的路径
    if not os.path.exists(config.save_dir):
        config.save_dir = '/SSD/lz/tsm_ptms_anomaly_detection/result/'

    print("save_dir = ", config.save_dir)  # 输出检查

    train_loader, train_set = get_loader_segment(config.index, config.data_path + config.dataset, batch_size=config.batch_size,
                                                 win_size=config.win_size, mode='train', dataset=config.dataset)
    val_loader, val_set = get_loader_segment(config.index, config.data_path + config.dataset, batch_size=config.batch_size,
                                             win_size=config.win_size, mode='val', dataset=config.dataset)
    test_loader, test_set = get_loader_segment(config.index, config.data_path + config.dataset, batch_size=config.batch_size,
                                               win_size=config.win_size, mode='test', dataset=config.dataset)
    train_set = train_set.train
    config.input_c = train_set.shape[-1]
    config.output_c = train_set.shape[-1]

    args = vars(config)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config, train_set, train_loader, val_set, val_loader, test_set, test_loader, config.cuda)
