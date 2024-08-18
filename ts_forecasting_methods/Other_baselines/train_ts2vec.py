import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import numpy as np
import argparse
import os
import random
import time
import datetime
from ts2vec.ts2vec import TS2Vec
import  ts2vec.tasks as tasks
import  ts2vec.datautils as datautils
from ts2vec.utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from Other_baselines.data_provider.data_factory import data_provider
from ts2vec.tasks import _eval_protocols as eval_protocols


def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])


def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }


def eval_forecasting_new(model, train_data, valid_data, test_data, pred_lens):
    padding = 200

    t = time.time()
    train_repr = model.encode(
        train_data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=32
    )
    valid_repr = model.encode(
        valid_data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=32
    )
    test_repr = model.encode(
        test_data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=32
    )
    ts2vec_infer_time = time.time() - t

    print("train_data.shape = ", train_data.shape)

    # train_repr = all_repr[:, train_slice]
    # valid_repr = all_repr[:, valid_slice]
    # test_repr = all_repr[:, test_slice]
    # print("train_repr.shape = ", train_repr.shape, ", valid_repr.shape = ", valid_repr.shape, ", test_repr.shape = ",
    #       test_repr.shape)
    #
    # print("n_covariate_cols = ", n_covariate_cols)
    #
    # train_data = data[:, train_slice, n_covariate_cols:]
    # valid_data = data[:, valid_slice, n_covariate_cols:]
    # test_data = data[:, test_slice, n_covariate_cols:]

    print("train_data.shape = ", train_data.shape, ", valid_data.shape = ", valid_data.shape, ", test_data.shape = ",
          test_data.shape)

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        # if test_data.shape[0] > 1:
        #     test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
        #     test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        # else:
        #     print("test_pred.shape = ", test_pred.shape, ", test_labels.shape = ", test_labels.shape)
        #     print("test_pred.swapaxes(0, 3).shape = ", test_pred.swapaxes(0, 3).shape)
        #     test_pred_inv = scaler.inverse_transform(test_pred)
        #     test_labels_inv = scaler.inverse_transform(test_labels)

        out_log[pred_len] = {
            'norm': test_pred,
            # 'raw': test_pred_inv,
            'norm_gt': test_labels,
            # 'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            # 'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }

    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res


def save_checkpoint_callback(
        save_every=1,
        unit='epoch'
):
    assert unit in ('epoch', 'iter')

    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')

    return callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset', default='traffic',
                        help='The dataset name')  ## 'ETTh1', 'ETTh2', 'electricity'  ETTm1
    # parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--run_name', default='ts2Vec',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    # parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, '
    #                                                               'UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--loader', type=str, default='forecast_csv',
                        help='The data loader used to load the experimental data.')  ## forecast_csv forecast_csv_univar
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=6,
                        help='The maximum allowed number of threads used by this process')
    # parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--eval', default=True,
                        help='Whether to perform evaluation after training')  ## action="store_true"
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')

    parser.add_argument('--save_dir', type=str, default='/dev_data/lz/ts_forecasting_methods/result/')
    parser.add_argument('--save_csv_name', type=str, default='ts2vec_forecasting_0730.csv')




    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/dev_data/lz/ts_forecasting_methods/ts2vec/datasets',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='subset for M4')  ## Hourly Daily Weekly Monthly Quarterly Yearly
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')

    args = parser.parse_args()

    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    # 检查路径是否存在，如果不存在则赋值为新的路径
    if not os.path.exists(args.save_dir):
        args.save_dir = '/SSD/lz/ts_forecasting_methods/result/'

    print("save_dir = ", args.save_dir)  # 输出检查

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)

    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)

    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        # data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(
        #     args.dataset)
        # train_data = data[:, train_slice]
        #
        # _train_data = data[:, train_slice, n_covariate_cols:]
        # valid_data = data[:, valid_slice, n_covariate_cols:]
        # test_data = data[:, test_slice, n_covariate_cols:]
        #
        #
        #
        #
        # print("data.shape = ", data.shape)
        #
        # print("train_slice = ", train_slice, ", valid_slice = ", valid_slice, ", test_slice = ", test_slice)
        #
        # print("train_data.shape = ", _train_data.shape, ", valid_data.shape = ", valid_data.shape,
        #       ", test_data.shape = ",
        #       test_data.shape)

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

        new_train_data = train_data.data_x[np.newaxis, :, :]
        new_vali_data = vali_data.data_x[np.newaxis, :, :]
        new_test_data = test_data.data_x[np.newaxis, :, :]

        print("new_train_data = ", new_train_data.shape, new_vali_data.shape, new_test_data.shape)




    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(
            args.dataset, univar=True)
        train_data = data[:, train_slice]
        print("raw data.shape = ", data.shape, ", train_data.shape = ", train_data.shape)

    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(
            args.dataset)
        train_data = data[:, train_slice]



    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(
            args.dataset, univar=True)
        train_data = data[:, train_slice]

    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(
            args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)

    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(
            args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')

    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    print("output_dims=args.repr_dims = ", args.repr_dims, ", input_dims = ", new_train_data.shape[-1])
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = TS2Vec(
        input_dims=new_train_data.shape[-1],
        device=device,
        **config
    )
    loss_log = model.fit(
        new_train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels,
                                                      eval_protocol='svm')
        elif task_type == 'forecasting':
            print("")
            # out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens,
            #                                        n_covariate_cols)
            pred_lens = [96, 192, 336, 720]
            if args.dataset == 'national_illness':
                pred_lens = [24, 36, 48, 60]
            out, eval_res = eval_forecasting_new(model, new_train_data, new_vali_data, new_test_data, pred_lens)

            # print("ts2vec out = ", out)
            print("ts2vec eval_res = ", eval_res)

            end_result = {}
            end_result['dataset'] = args.dataset
            end_result['random_seed'] = args.random_seed
            for _pred in pred_lens:
                _MSE = str(_pred) + "_MSE"
                end_result[_MSE] = eval_res['ours'][_pred]['norm']['MSE']
                _MAE = str(_pred) + "_MAE"
                end_result[_MAE] = eval_res['ours'][_pred]['norm']['MAE']




            # end_result['24_MSE'] = eval_res['ours'][24]['norm']['MSE']
            # end_result['24_MAE'] = eval_res['ours'][24]['norm']['MAE']
            #
            # end_result['48_MSE'] = eval_res['ours'][48]['norm']['MSE']
            # end_result['48_MAE'] = eval_res['ours'][48]['norm']['MAE']
            #
            # if args.dataset == 'ETTm1':
            #     end_result['168_MSE'] = eval_res['ours'][96]['norm']['MSE']
            #     end_result['168_MAE'] = eval_res['ours'][96]['norm']['MAE']
            #
            #     end_result['336_MSE'] = eval_res['ours'][288]['norm']['MSE']
            #     end_result['336_MAE'] = eval_res['ours'][288]['norm']['MAE']
            #
            #     end_result['720_MSE'] = eval_res['ours'][672]['norm']['MSE']
            #     end_result['720_MAE'] = eval_res['ours'][672]['norm']['MAE']
            # else:
            #
            #     end_result['168_MSE'] = eval_res['ours'][168]['norm']['MSE']
            #     end_result['168_MAE'] = eval_res['ours'][168]['norm']['MAE']
            #
            #     end_result['336_MSE'] = eval_res['ours'][336]['norm']['MSE']
            #     end_result['336_MAE'] = eval_res['ours'][336]['norm']['MAE']
            #
            #     end_result['720_MSE'] = eval_res['ours'][720]['norm']['MSE']
            #     end_result['720_MAE'] = eval_res['ours'][720]['norm']['MAE']

            import pandas as pd

            # 转换字典为 DataFrame
            # df = pd.DataFrame([eval_res])
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





        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps,
                                                         all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels,
                                                                   all_train_timestamps, all_test_data, all_test_labels,
                                                                   all_test_timestamps, delay)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

    print("Finished.")
