import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import os
import torch
from other_anomaly_baselines.exp_anomaly_detection import Exp_Anomaly_Detection
from other_anomaly_baselines.datasets.data_loader import get_loader_segment
import numpy as np
import random
import datautils
from torch.utils.data import TensorDataset, DataLoader


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


if __name__ == '__main__':
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='anomaly_detection',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='TimesNet',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    # parser.add_argument('--data', type=str, default='UCR', help='dataset type')   ##  SMD, MSL, SMAP, PSM, SWAT, NIPS_TS_Swan, UCR, NIPS_TS_Water
    parser.add_argument('--data', type=str, default='kpi')  ##  kpi, yahoo
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=1, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') ## 55 for MSL, 38 for SMD, SMAP for 25, PSM for 25, SWAT for 51, NIPS_TS_Swan for 38,

    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')  ## NIPS_TS_Water for 38, UCR for 1
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=8, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=16, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # Default
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--data_path', type=str, default='datasets/')
    parser.add_argument('--win_size', type=int, default=100)

    parser.add_argument('--save_dir', type=str, default='/dev_data/lz/tsm_ptms_anomaly_detection/result/')
    parser.add_argument('--save_csv_name', type=str, default='timesnet_uni_0722.csv')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # 检查路径是否存在，如果不存在则赋值为新的路径
    if not os.path.exists(args.save_dir):
        args.save_dir = '/SSD/lz/tsm_ptms_anomaly_detection/result/'

    print("save_dir = ", args.save_dir)  # 输出检查

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Anomaly_Detection

    # dataset = 'MSL'
    # _train_loader, _train_set = get_loader_segment(args.index, args.data_path + dataset,
    #                                                batch_size=args.batch_size,
    #                                                win_size=args.win_size, mode='train', dataset=dataset)
    #
    # _train_set = _train_set.train
    #
    # print("_train_set.shape = ", _train_set.shape)

    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(
        args.data)
    train_data = datautils.gen_ano_train_data(all_train_data)

    print("train_data.shape = ", train_data.shape)
    _train_data = train_data[0]
    print("000train_data.shape = ", train_data.shape, type(train_data))
    _train_data = np.array(_train_data)
    print("111_train_data.shape = ", _train_data.shape, type(_train_data))

    train_dataset = UniLoader(_train_data, args.win_size, 1)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2,
                              drop_last=True)


    val_loader = train_loader

    args.input_c = train_data.shape[-1]
    args.output_c = train_data.shape[-1]




    # train_loader, train_set = get_loader_segment(args.index, args.data_path + args.data, batch_size=args.batch_size,
    #                                           win_size=args.win_size, mode='train', dataset=args.data)
    # val_loader, val_set = get_loader_segment(args.index, args.data_path + args.data, batch_size=args.batch_size,
    #                                          win_size=args.win_size, mode='val', dataset=args.data)
    # test_loader, test_set = get_loader_segment(args.index, args.data_path + args.data, batch_size=args.batch_size,
    #                                          win_size=args.win_size, mode='test', dataset='UCR')

    # train_set = train_set.train
    # val_set = val_set.val
    # test_set = test_set.test

    print("train_set.shape = ", _train_data.shape)
    args.enc_in = _train_data.shape[-1]
    args.c_out = _train_data.shape[-1]

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args, _train_data, train_loader, _train_data, train_loader, _train_data, train_loader)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train_uni(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            eval_res = exp.test_uni(setting, all_train_data, all_test_data, all_test_labels, all_test_timestamps, delay, args)
            torch.cuda.empty_cache()

            print("result_dict = ", eval_res)

            eval_res['dataset'] = args.data + str(args.index)
            import pandas as pd

            # 转换字典为 DataFrame
            df = pd.DataFrame([eval_res])
            # 指定保存路径
            save_path = args.save_dir + args.save_csv_name

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

    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
