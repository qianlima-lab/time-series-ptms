from data_provider.data_factory import data_provider


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

parser.add_argument('--freq', type=str, default='m',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
# parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--root_path', type=str, default='./datasets/')
parser.add_argument('--data_path', type=str, default='national_illness.csv')   ## ETTh1 ETTh2 ETTm1  ETTm2 electricity traffic weather exchange_rate national_illness
parser.add_argument('--data', type=str, default='custom')  ## ett_h  ett_m custom

# 'custom': Dataset_Custom,
#     'tsf_data': Dataset_TSF,
#     'ett_h': Dataset_ETT_hour,
#     'ett_m': Dataset_ETT_minute,

parser.add_argument('--features', type=str, default='M')
# parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

# parser.add_argument('--seq_len', type=int, default=512)
# parser.add_argument('--pred_len', type=int, default=96)
# parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)



args = parser.parse_args()

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

train_data, train_loader = data_provider(args, 'train')
vali_data, vali_loader = data_provider(args, 'val')
test_data, test_loader = data_provider(args, 'test')

print("dataset name = ", args.data_path)

print("type train_data = ", type(train_data))

print("train_data = ", train_data)
print(train_data.data_x.shape, train_data.data_y.shape)

print("train_data = ", train_data)
print(vali_data.data_x.shape, vali_data.data_y.shape)

print("train_data = ", train_data)
print(test_data.data_x.shape, test_data.data_y.shape)


#
# print("train_data.shape = ", train_data.shape)
# print("vali_data.shape = ", vali_data.shape)
# print("test_data.shape = ", test_data.shape)