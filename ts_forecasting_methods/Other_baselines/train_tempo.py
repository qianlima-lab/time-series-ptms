import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from Other_baselines.data_provider.data_factory_tempo import data_provider
from Other_baselines.utils.tools_tempo import EarlyStopping, adjust_learning_rate, vali, test
from torch.utils.data import Subset
from tqdm import tqdm
from Other_baselines.models.PatchTST import PatchTST
from Other_baselines.models.GPT4TS import GPT4TS
from Other_baselines.models.TEMPO import TEMPO

import torch
import torch.nn as nn
from numpy.random import choice

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random
import sys

from omegaconf import OmegaConf


def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config


warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser(description='GPT4TS')
parser.add_argument('--random_seed', type=int, default=42, help='random seed')
parser.add_argument('--model_id', type=str, default='weather_GTP4TS_multi-debug')
parser.add_argument('--checkpoints', type=str, default='/SSD/lz/ts_forecasting_methods/Other_baselines/checkpoints_multi_dataset/')
parser.add_argument('--task_name', type=str, default='long_term_forecast')

parser.add_argument('--stl_weight', type=float, default=0.01)
parser.add_argument('--config_path', type=str, default='/SSD/lz/ts_forecasting_methods/Other_baselines/data_config.yml')
parser.add_argument('--datasets', type=str, default='exchange')
parser.add_argument('--target_data', type=str, default='exchange')
# python train_tempo.py --datasets exchange --target_data exchange --data custom --data_path exchange_rate.csv --random_seed 42;
# data loader
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
# parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--root_path', type=str, default='/SSD/lz/ts_forecasting_methods/ts2vec/datasets',
                    help='root path of the data file')
parser.add_argument('--data_path', type=str, default='exchange_rate.csv', help='data file')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")


parser.add_argument('--prompt', type=int, default=0)
parser.add_argument('--num_nodes', type=int, default=1)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.9)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type3')  # for what
parser.add_argument('--patience', type=int, default=5)

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='TEMPO')  ### GPT4TS_multi TEMPO
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--equal', type=int, default=1, help='1: equal sampling, 0: dont do the equal sampling')
parser.add_argument('--pool', action='store_true', help='whether use prompt pool')
parser.add_argument('--no_stl_loss', action='store_true', help='whether use prompt pool')


parser.add_argument('--use_token', type=int, default=0)
parser.add_argument('--electri_multiplier', type=int, default=1)
parser.add_argument('--traffic_multiplier', type=int, default=1)
parser.add_argument('--embed', type=str, default='timeF')

parser.add_argument('--save_dir', type=str, default='/SSD/lz/ts_forecasting_methods/result/')
parser.add_argument('--save_csv_name', type=str, default='tempo_forecasting_0729.csv')

# args = parser.parse_args([])
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    args.root_path = '/dev_data/lz/ts_forecasting_methods/ts2vec/datasets'
    args.save_dir = '/dev_data/lz/ts_forecasting_methods/result/'
    args.config_path = '/dev_data/lz/ts_forecasting_methods/Other_baselines/data_config.yml'
    args.checkpoints = '/dev_data/lz/ts_forecasting_methods/Other_baselines/checkpoints_multi_dataset/'


config = get_init_config(args.config_path)

# fix_seed = 2021
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

args.itr = 1

print(args)

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

mses = []
maes = []
for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len,
                                                                             args.pred_len,
                                                                             args.d_model, args.n_heads, args.e_layers,
                                                                             args.gpt_layers,
                                                                             args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    print("path = ", path)
    if not os.path.exists(path):
        os.makedirs(path)

    # if args.freq == 0:
    #     args.freq = 'h'

    device = torch.device('cuda:0')
    if args.gpu == 1:
        device = torch.device('cuda:1')

    train_data_name = args.datasets.split(',')
    print(train_data_name)
    train_datas = []
    val_datas = []
    min_sample_num = sys.maxsize
    for dataset_singe in args.datasets.split(','):
        print(dataset_singe)
        # args.data = config['datasets'][dataset_singe].data
        # args.root_path = config['datasets'][dataset_singe].root_path
        # args.data_path = config['datasets'][dataset_singe].data_path
        args.data_name = config['datasets'][dataset_singe].data_name
        args.features = config['datasets'][dataset_singe].features
        args.freq = config['datasets'][dataset_singe].freq
        args.target = config['datasets'][dataset_singe].target
        args.embed = config['datasets'][dataset_singe].embed
        args.percent = config['datasets'][dataset_singe].percent
        args.lradj = config['datasets'][dataset_singe].lradj

        print("args.data_name = ", args.data_name)
        print("args.features = ",  args.features)
        print("args.freq = ", args.freq)
        print("args.target = ", args.target)
        print("args.embed = ", args.embed)

        if args.freq == 0:
            args.freq = 'h'

        print("dataset: ", args.data)
        train_data, train_loader = data_provider(args, 'train')
        if dataset_singe not in ['ETTh1', 'ETTh2', 'ILI', 'exchange']:
            min_sample_num = min(min_sample_num, len(train_data))

        # args.percent = 20
        vali_data, vali_loader = data_provider(args, 'val')
        # args.percent = 100

        # train_datas.append(train_data)
        val_datas.append(vali_data)

    for dataset_singe in args.datasets.split(','):
        print(dataset_singe)
        # args.data = config['datasets'][dataset_singe].data
        # args.root_path = config['datasets'][dataset_singe].root_path
        # args.data_path = config['datasets'][dataset_singe].data_path
        args.data_name = config['datasets'][dataset_singe].data_name
        args.features = config['datasets'][dataset_singe].features
        args.freq = config['datasets'][dataset_singe].freq
        args.target = config['datasets'][dataset_singe].target
        args.embed = config['datasets'][dataset_singe].embed
        args.percent = config['datasets'][dataset_singe].percent
        args.lradj = config['datasets'][dataset_singe].lradj
        if args.freq == 0:
            args.freq = 'h'
        # if args.freq != 'h':
        #     args.freq = SEASONALITY_MAP[test_data.freq]
        #     print("freq = {}".format(args.freq))

        print("dataset: ", args.data)
        train_data, train_loader = data_provider(args, 'train')
        if dataset_singe not in ['ETTh1', 'ETTh2', 'ILI', 'exchange'] and args.equal == 1:
            train_data = Subset(train_data, choice(len(train_data), min_sample_num))
        if args.electri_multiplier > 1 and args.equal == 1 and dataset_singe in ['electricity']:
            train_data = Subset(train_data, choice(len(train_data), int(min_sample_num * args.electri_multiplier)))
        if args.traffic_multiplier > 1 and args.equal == 1 and dataset_singe in ['traffic']:
            train_data = Subset(train_data, choice(len(train_data), int(min_sample_num * args.traffic_multiplier)))
        train_datas.append(train_data)

    if len(train_datas) > 1:
        train_data = torch.utils.data.ConcatDataset([train_datas[0], train_datas[1]])
        vali_data = torch.utils.data.ConcatDataset([val_datas[0], val_datas[1]])
        for i in range(2, len(train_datas)):
            train_data = torch.utils.data.ConcatDataset([train_data, train_datas[i]])

            vali_data = torch.utils.data.ConcatDataset([vali_data, val_datas[i]])

        # import pdb; pdb.set_trace()
        print("Way1", len(train_data))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
        vali_loader = torch.utils.data.DataLoader(vali_data, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers)

        # args.data = config['datasets'][args.target_data].data
        # args.root_path = config['datasets'][args.target_data].root_path
        # args.data_path = config['datasets'][args.target_data].data_path
        args.data_name = config['datasets'][args.target_data].data_name
        args.features = config['datasets'][dataset_singe].features
        args.freq = config['datasets'][args.target_data].freq
        args.target = config['datasets'][args.target_data].target
        args.embed = config['datasets'][args.target_data].embed
        args.percent = config['datasets'][args.target_data].percent
        args.lradj = config['datasets'][args.target_data].lradj
        if args.freq == 0:
            args.freq = 'h'
        test_data, test_loader = data_provider(args, 'test')

    time_now = time.time()
    train_steps = len(train_loader)  # 190470 -52696

    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'TEMPO':
        model = TEMPO(args, device)
        model.to(device)
    else:
        model = GPT4TS(args, device)
    # mse, mae = test(model, test_data, test_loader, args, device, ii)

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()

            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))


        criterion = SMAPE()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    for epoch in range(args.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        print("len(train_loader) = ", len(train_loader))
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in tqdm(
                enumerate(train_loader), total=len(train_loader)):

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            seq_trend = seq_trend.float().to(device)
            seq_seasonal = seq_seasonal.float().to(device)
            seq_resid = seq_resid.float().to(device)

            # print(seq_seasonal.shape)
            if args.model == 'TEMPO' or 'multi' in args.model:
                outputs, loss_local = model(batch_x, ii, seq_trend, seq_seasonal,
                                            seq_resid)  # + model(seq_seasonal, ii) + model(seq_resid, ii)
            elif 'former' in args.model:
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, ii)
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y)
            if args.model == 'GPT4TS_multi' or args.model == 'TEMPO_t5':
                if not args.no_stl_loss:
                    loss += args.stl_weight * loss_local
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path), strict=False)
    print("------------------------------------")
    test_data, test_loader = data_provider(args, 'test')
    mse, mae = test(model, test_data, test_loader, args, device, ii)
    torch.cuda.empty_cache()
    print('test on the ' + str(args.target_data) + ' dataset: mse:' + str(mse) + ' mae:' + str(mae))

    end_result = {}
    end_result['dataset'] = args.data_path
    end_result['pred_len'] = args.pred_len
    end_result['random_seed'] = args.random_seed
    end_result['MSE'] = mse
    end_result['MAE'] = mae

    import pandas as pd

    # 指定保存路径
    save_path = args.save_dir + args.save_csv_name

    # 转换字典为 DataFrame
    df_new = pd.DataFrame([end_result])

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

    print("Save success!!!")

    mses.append(mse)
    maes.append(mae)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
