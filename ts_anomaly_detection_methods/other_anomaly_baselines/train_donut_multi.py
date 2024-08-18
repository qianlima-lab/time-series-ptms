import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from donut import DONUT
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
    parser.add_argument('--dataset', default='PSM',
                        help='The dataset name, yahoo, kpi')  ##  SMD, MSL, SMAP, PSM, SWAT, NIPS_TS_Swan, UCR, NIPS_TS_Water
    parser.add_argument('--is_multi', default=True, help='The dataset name, yahoo, kpi')
    parser.add_argument('--datapath', default='./datasets/', help='')
    parser.add_argument('--index', type=int, default=203, help='')
    parser.add_argument('--run_name', default='donut',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    # parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data--anomaly')
    parser.add_argument('--loader', type=str, default='anomaly',
                        help='The data loader used to load the experimental data--anomaly')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--latent_dim', type=int, default=100, help='The units of the hidden layer.')
    parser.add_argument('--hidden_dim', type=int, default=3, help='The dims of the hidden representation (z).')
    parser.add_argument('--z_kld_weight', type=float, default=1)
    parser.add_argument('--x_kld_weight', type=float, default=1)
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', default=True, help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')

    parser.add_argument('--save_dir', type=str, default='/dev_data/lz/tsm_ptms_anomaly_detection/result/')
    parser.add_argument('--save_csv_name', type=str, default='donut_ucr_0727.csv')

    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    # 检查路径是否存在，如果不存在则赋值为新的路径
    if not os.path.exists(args.save_dir):
        args.save_dir = '/SSD/lz/tsm_ptms_anomaly_detection/result/'

    print("save_dir = ", args.save_dir)  # 输出检查

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print('Loading data... ', end='')
    if args.loader == 'anomaly':
        task_type = 'anomaly_detection'

        if args.is_multi:
            from datasets.data_loader import get_loader_segment

            data_path = args.datapath + args.dataset + '/'
            print("data_path = ", data_path)
            _, train_data_loader = get_loader_segment(args.index, data_path, args.batch_size, win_size=100, step=100,
                                                      mode='train',
                                                      dataset=args.dataset)

            all_train_data = train_data_loader.train
            all_train_labels = None
            all_train_timestamps = None
            all_test_data = train_data_loader.test
            all_test_labels = train_data_loader.test_labels
            all_test_timestamps = None
            delay = 5

            print("all_train_data test_data, test_labels.shape = ", all_train_data.shape, all_test_data.shape,
                  all_test_labels.shape)
            train_data = np.expand_dims(all_train_data, axis=0)
            print("train_data.shape = ", train_data.shape)
            print("Read Success!!!")
        else:
            all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(
                args.dataset)
            train_data = datautils.gen_ano_train_data(all_train_data)
    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    if args.irregular > 0:
        raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        z_kld_weight=args.z_kld_weight,
        x_kld_weight=args.x_kld_weight,
        max_train_length=args.max_train_length
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = DONUT(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    loss_log = model.train(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}")
    print("Training time(seconds): ", t)

    if args.eval:
        if task_type == 'anomaly_detection':
            out, eval_res = model.evaluate(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data,
                                           all_test_labels, all_test_timestamps, delay, is_multi=args.is_multi, ucr_index=args.index)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

        eval_res['dataset'] = args.dataset + str(args.index)
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

    print("Finished.")
