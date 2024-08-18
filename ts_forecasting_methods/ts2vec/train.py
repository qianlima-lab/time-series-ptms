import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

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
    parser.add_argument('--dataset', default='ETTh1', help='The dataset name')  ## 'ETTh1', 'ETTh2', 'electricity'  ETTm1
    # parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--run_name', default='ts2Vec',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    # parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, '
    #                                                               'UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--loader', type=str, default='forecast_csv',
                        help='The data loader used to load the experimental data.')  ## forecast_csv forecast_csv_univar
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=6, help='The maximum allowed number of threads used by this process')
    # parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--eval', default=True,
                        help='Whether to perform evaluation after training')  ## action="store_true"
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')

    parser.add_argument('--save_dir', type=str, default='/dev_data/lz/ts_forecasting_methods/result/')
    parser.add_argument('--save_csv_name', type=str, default='ts2vec_forecasting_0724.csv')

    args = parser.parse_args()
    
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
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        print("raw data.shape = ", data.shape, ", train_data.shape = ", train_data.shape)
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
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
    print("output_dims=args.repr_dims = ", args.repr_dims, ", input_dims = ", train_data.shape[-1])
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
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            print("")
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)

            # print("ts2vec out = ", out)
            print("ts2vec eval_res = ", eval_res)

            end_result = {}
            end_result['dataset'] = args.dataset
            end_result['24_MSE'] = eval_res['ours'][24]['norm']['MSE']
            end_result['24_MAE'] = eval_res['ours'][24]['norm']['MAE']

            end_result['48_MSE'] = eval_res['ours'][48]['norm']['MSE']
            end_result['48_MAE'] = eval_res['ours'][48]['norm']['MAE']

            if args.dataset == 'ETTm1':
                end_result['168_MSE'] = eval_res['ours'][96]['norm']['MSE']
                end_result['168_MAE'] = eval_res['ours'][96]['norm']['MAE']

                end_result['336_MSE'] = eval_res['ours'][288]['norm']['MSE']
                end_result['336_MAE'] = eval_res['ours'][288]['norm']['MAE']

                end_result['720_MSE'] = eval_res['ours'][672]['norm']['MSE']
                end_result['720_MAE'] = eval_res['ours'][672]['norm']['MAE']
            else:

                end_result['168_MSE'] = eval_res['ours'][168]['norm']['MSE']
                end_result['168_MAE'] = eval_res['ours'][168]['norm']['MAE']

                end_result['336_MSE'] = eval_res['ours'][336]['norm']['MSE']
                end_result['336_MAE'] = eval_res['ours'][336]['norm']['MAE']

                end_result['720_MSE'] = eval_res['ours'][720]['norm']['MSE']
                end_result['720_MAE'] = eval_res['ours'][720]['norm']['MAE']

            import pandas as pd

            # 转换字典为 DataFrame
            df = pd.DataFrame([eval_res])
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
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

    print("Finished.")
