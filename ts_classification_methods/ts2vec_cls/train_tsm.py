import argparse
import datetime
import os
import sys
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import torch

from data.preprocessing import normalize_per_series, fill_nan_value, normalize_train_val_test
from ts2vec_cls import tasks
from ts2vec_cls.ts2vec import TS2Vec
from ts2vec_cls.utils import init_dl_program, name_with_datetime
from tsm_utils import build_dataset, save_cls_result, get_all_datasets, set_seed


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
    parser.add_argument('--dataset', type=str, default='Coffee', help='The dataset name')
    parser.add_argument('--dataroot', type=str, default='/SSD/lz/UCRArchive_2018',
                        help='path of UCR folder')  ## '/SSD/lz/UCRArchive_2018', None
    parser.add_argument('--run_name', default='UCR',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, default='UCR',
                        help='The data loader used to load the experimental data. This can be set to UCR, UEA, '
                             'forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=1,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped '
                             'into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--random_seed', type=int, default=42, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=8,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", default=True,
                        help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--normalize_way', type=str, default='single', help='single or train_set')
    parser.add_argument('--save_csv_name', type=str, default='ts2vec_test_cls_0409_')
    parser.add_argument('--save_dir', type=str, default='/SSD/lz/time_tsm/ts2vec_cls/result')
    args = parser.parse_args()
    set_seed(args)

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print('Loading data... ', end='')

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

    sum_dataset, sum_target, num_classes = build_dataset(args)

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target)

    test_accuracies = []
    train_time = 0.0
    for i, train_dataset in enumerate(train_datasets):
        print("\nStart K_fold = ", i)
        train_labels = train_targets[i]

        val_dataset = val_datasets[i]
        val_labels = val_targets[i]

        test_dataset = test_datasets[i]
        test_labels = test_targets[i]

        # mean impute for missing values in dataset
        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        if args.normalize_way == 'single':
            # TODO normalize per series
            train_dataset = normalize_per_series(train_dataset)
            val_dataset = normalize_per_series(val_dataset)
            test_dataset = normalize_per_series(test_dataset)
        else:
            train_dataset, val_dataset, test_dataset = normalize_train_val_test(train_dataset, val_dataset,
                                                                                test_dataset)

        train_dataset = train_dataset[..., np.newaxis]
        val_dataset = val_dataset[..., np.newaxis]
        test_dataset = test_dataset[..., np.newaxis]

        # print(type(train_dataset))
        train_val_dataset = np.concatenate((train_dataset, val_dataset))
        train_val_labels = np.concatenate((train_labels, val_labels))
        # print(train_labels.shape, val_labels.shape)
        # print("train, val train_val_data.shape = ", train_dataset.shape, val_dataset.shape, train_val_dataset.shape, train_val_labels.shape)

        t = time.time()

        model = TS2Vec(
            input_dims=train_dataset.shape[-1],
            device=device,
            **config
        )
        loss_log = model.fit(
            train_dataset,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=True
        )
        # model.save(f'{run_dir}/model.pkl')
        ## evalution on test_dataset,
        out, eval_res = tasks.eval_classification(model, train_val_dataset, train_val_labels, test_dataset, test_labels,
                                                  eval_protocol='svm')
        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
        train_time += t

        print('Evaluation result:', eval_res)
        test_accuracies.append(eval_res['acc'])

    test_accuracies = torch.Tensor(test_accuracies)
    save_cls_result(args, test_accu=torch.mean(test_accuracies), test_std=torch.std(test_accuracies),
                    train_time=train_time / 5, end_val_epoch=0.00, seeds=args.random_seed)

    print("Finished.")
