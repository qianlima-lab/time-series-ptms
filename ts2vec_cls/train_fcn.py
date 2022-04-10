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
from torch.utils.data import DataLoader

from data.dataloader import UCRDataset
from data.preprocessing import normalize_per_series, fill_nan_value, normalize_train_val_test
from ts2vec_cls.ts2vec import TS2Vec
from ts2vec_cls.utils import init_dl_program, name_with_datetime
from tsm_utils import build_dataset, build_model, build_loss, evaluate, save_cls_result, get_all_datasets, set_seed


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
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--task', type=str, default='classification', help='classification or reconstruction')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn or dilated')
    parser.add_argument('--classifier', type=str, default='nonlinear', help='type of classifier(linear or nonlinear)')
    parser.add_argument('--run_name', default='UCR',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, default='UCR',
                        help='The data loader used to load the experimental data. This can be set to UCR, UEA, '
                             'forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=1,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--fcn_batch_size', type=int, default=128,
                        help='(16, 128) larger batch size on the big dataset, ')  # 16
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped '
                             'into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--fcn_epoch', type=int, default=1000, help='fcn training epoch')
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
    parser.add_argument('--save_csv_name', type=str, default='ts2vec_test_fcncls_0404_')
    parser.add_argument('--save_dir', type=str, default='/SSD/lz/time_tsm/ts2vec_cls/result')
    parser.add_argument('--cuda', type=str, default='cuda:1')

    args = parser.parse_args()
    set_seed(args)

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    device_fcn = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

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
    args.num_classes = num_classes
    if sum_dataset.shape[0] < args.fcn_batch_size:
        args.fcn_batch_size = 16

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target)

    test_accuracies = []
    end_val_epochs = []
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

        print("train_data.shape = ", train_dataset.shape)

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
        train_repr = model.encode(train_dataset, encoding_window='full_series' if train_labels.ndim == 1 else None)
        val_repr = model.encode(val_dataset, encoding_window='full_series' if val_labels.ndim == 1 else None)
        test_repr = model.encode(test_dataset, encoding_window='full_series' if test_labels.ndim == 1 else None)
        # accu = test_accu.cpu().numpy()
        print("data info = ", train_repr.shape, test_repr.shape, train_dataset.shape, test_dataset.shape)
        # print(type(train_repr), train_repr[:2])
        model_fcn, classifier = build_model(args)
        model_fcn, classifier = model_fcn.to(device_fcn), classifier.to(device_fcn)
        loss = build_loss(args).to(device_fcn)
        optimizer = torch.optim.Adam([{'params': model_fcn.parameters()}, {'params': classifier.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)

        train_set = UCRDataset(torch.from_numpy(train_repr).to(device_fcn),
                               torch.from_numpy(train_labels).to(device_fcn).to(torch.int64))
        val_set = UCRDataset(torch.from_numpy(val_repr).to(device_fcn),
                             torch.from_numpy(val_labels).to(device_fcn).to(torch.int64))
        test_set = UCRDataset(torch.from_numpy(test_repr).to(device_fcn),
                              torch.from_numpy(test_labels).to(device_fcn).to(torch.int64))

        train_loader = DataLoader(train_set, batch_size=args.fcn_batch_size, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.fcn_batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.fcn_batch_size, num_workers=0)

        train_loss = []
        train_accuracy = []
        num_steps = args.fcn_epoch // args.batch_size

        last_loss = float('inf')
        stop_count = 0
        increase_count = 0

        test_accuracy = 0
        min_val_loss = float('inf')
        end_val_epoch = 0

        num_steps = train_set.__len__() // args.batch_size
        for epoch in range(args.fcn_epoch):
            # early stopping in finetune
            if stop_count == 50 or increase_count == 50:
                print('model convergent at epoch {}, early stopping'.format(epoch))
                break

            epoch_train_loss = 0
            epoch_train_acc = 0
            model_fcn.train()
            classifier.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = model_fcn(x)
                pred = classifier(pred)

                step_loss = loss(pred, y)
                step_loss.backward()
                optimizer.step()

                epoch_train_loss += step_loss.item()
                epoch_train_acc += torch.sum(torch.argmax(pred.data, axis=1) == y) / len(y)

            epoch_train_loss /= num_steps
            epoch_train_acc /= num_steps

            model_fcn.eval()
            classifier.eval()
            val_loss, val_accu = evaluate(val_loader, model_fcn, classifier, loss, device_fcn)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                end_val_epoch = epoch
                test_loss, test_accuracy = evaluate(test_loader, model_fcn, classifier, loss, device_fcn)

            if epoch % 100 == 0:
                print(
                    "epoch : {}, train loss: {} , train accuracy : {}, \nval loss : {}, val accuracy : {}, \ntest loss : {}, test accuracy : {}".format(
                        epoch, epoch_train_loss, epoch_train_acc, val_loss, val_accu, test_loss, test_accuracy))

            if abs(last_loss - val_loss) <= 1e-4:
                stop_count += 1
            else:
                stop_count = 0

            if val_loss > last_loss:
                increase_count += 1
            else:
                increase_count = 0

            last_loss = val_loss

        # out, eval_res = tasks.eval_classification(model, train_dataset, train_labels, test_dataset, test_labels,
        #                                           eval_protocol='svm')
        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
        train_time += t

        # print('Evaluation result:', eval_res)
        # test_accuracies.append(eval_res['acc'])
        test_accuracies.append(test_accuracy)
        end_val_epochs.append(end_val_epoch)

    test_accuracies = torch.Tensor(test_accuracies)
    end_val_epochs = np.array(end_val_epochs)
    save_cls_result(args, test_accu=torch.mean(test_accuracies), test_std=torch.std(test_accuracies),
                    train_time=train_time / 5, end_val_epoch=np.mean(end_val_epochs))

    print("Finished.")
