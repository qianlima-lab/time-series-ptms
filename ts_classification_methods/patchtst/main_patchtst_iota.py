import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import argparse
import time

import numpy as np
import torch

from torch.utils.data import DataLoader

from gpt4ts.gpt4ts_utils import load_UEA, normalize_uea_set, UEADataset, save_cls_new_result, set_seed, fill_nan_value, get_all_datasets, build_loss

from gpt4ts.models.gpt4ts import gpt4ts

from patchtst.models.patchTST import PatchTST
from patchtst.patch_mask import PatchCB


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len

    xb = xb[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


def evaluate_gpt4ts(args, val_loader, model, loss):
    val_loss = 0
    val_accu = 0

    sum_len = 0
    for data, target in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        with torch.no_grad():
            xb, num_patch = create_patch(xb=data.permute(0, 2, 1), patch_len=args.patch_len, stride=args.stride)
            val_pred = model(xb)
            val_loss += loss(val_pred, target).item()
            val_accu += torch.sum(torch.argmax(val_pred.data, axis=1) == target)
            sum_len += len(target)

    return val_loss / sum_len, val_accu / sum_len


if __name__ == '__main__':  ##
    parser = argparse.ArgumentParser()

    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='LSST',
                        help='dataset(in ucr)')  # LSST Heartbeat Images
    # parser.add_argument('--dataroot', type=str, default='../UCRArchive_2018', help='path of UCR folder')
    # parser.add_argument('--dataroot', type=str, default='/dev_data/lz/time_series_pretrain/datasets/UCRArchive_2018',
    #                     help='path of UCR folder')
    # parser.add_argument('--dataroot', type=str, default='/SSD/lz/UCRArchive_2018', help='path of UCR folder')
    parser.add_argument('--dataroot', type=str, default='/dev_data/lz/Multivariate2018_arff', help='path of UEA folder')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--normalize_way', type=str, default='single', help='single or train_set')
    parser.add_argument('--seq_len', type=int, default=46, help='seq_len')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # parser.add_argument('--patch_size', type=int, default=8, help='patch_size')
    # parser.add_argument('--stride', type=int, default=8, help='stride')

    parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')

    # Patch
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride between patch')

    # RevIN
    parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
    # Model args
    parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
    parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
    parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
    parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')

    # Semi training
    parser.add_argument('--labeled_ratio', type=float, default='0.1', help='0.1, 0.2, 0.4')

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    parser.add_argument('--save_dir', type=str, default='/dev_data/lz/time_series_label_noise/result')
    parser.add_argument('--save_csv_name', type=str, default='patchtst_supervised_patch8_1224_')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='type of classifier(linear or nonlinear)')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    # sum_dataset, sum_target, num_classes = build_dataset(args)
    sum_dataset, sum_target, num_classes = load_UEA(args.dataroot, args.dataset)
    # args.num_classes = num_classes
    # args.seq_len = sum_dataset.shape[1]

    args.num_classes = num_classes
    args.seq_len = sum_dataset.shape[1]
    args.input_size = sum_dataset.shape[2]

    # get number of patches
    num_patch = (max(args.seq_len, args.patch_len) - args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    while sum_dataset.shape[0] * 0.6 < args.batch_size:
        args.batch_size = args.batch_size // 2

    print("args.batch_size = ", args.batch_size, ", sum_dataset.shape = ", sum_dataset.shape)

    # get model
    model = PatchTST(c_in=args.input_size,
                     target_dim=args.target_points,
                     patch_len=args.patch_len,
                     stride=args.stride,
                     num_patch=num_patch,
                     n_layers=args.n_layers,
                     n_heads=args.n_heads,
                     d_model=args.d_model,
                     shared_embedding=True,
                     d_ff=args.d_ff,
                     dropout=args.dropout,
                     head_dropout=args.head_dropout,
                     act='relu',
                     head_type='classification',
                     res_attention=False
                     )


    # model = gpt4ts(max_seq_len=args.seq_len, num_classes=args.num_classes, var_len=args.input_size, patch_size=args.patch_size, stride=args.stride)
    model = model.to(device)

    # model, classifier = build_model(args)
    # model, classifier = model.to(device), classifier.to(device)
    loss = build_loss(args).to(device)

    model_init_state = model.state_dict()
    # classifier_init_state = classifier.state_dict()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target)

    losses = []
    test_accuracies = []
    train_time = 0.0
    end_val_epochs = []

    for i, train_dataset in enumerate(train_datasets):
        t = time.time()
        model.load_state_dict(model_init_state)
        # classifier.load_state_dict(classifier_init_state)
        print('{} fold start training and evaluate'.format(i))

        train_target = train_targets[i]
        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        if args.normalize_way == 'single':
            # TODO normalize per series
            train_dataset = normalize_uea_set(train_dataset)
            val_dataset = normalize_uea_set(val_dataset)
            test_dataset = normalize_uea_set(test_dataset)
        # else:
        #     train_dataset, val_dataset, test_dataset = normalize_train_val_test(train_dataset, val_dataset,
        #                                                                         test_dataset)

        train_set = UEADataset(torch.from_numpy(train_dataset).type(torch.FloatTensor).to(device),
                               torch.from_numpy(train_target).type(torch.FloatTensor).to(device).to(torch.int64))
        val_set = UEADataset(torch.from_numpy(val_dataset).type(torch.FloatTensor).to(device),
                             torch.from_numpy(val_target).type(torch.FloatTensor).to(device).to(torch.int64))
        test_set = UEADataset(torch.from_numpy(test_dataset).type(torch.FloatTensor).to(device),
                              torch.from_numpy(test_target).type(torch.FloatTensor).to(device).to(torch.int64))

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

        train_loss = []
        train_accuracy = []
        num_steps = args.epoch // args.batch_size

        last_loss = float('inf')
        stop_count = 0
        increase_count = 0

        num_steps = train_set.__len__() // args.batch_size

        min_val_loss = float('inf')
        test_accuracy = 0
        end_val_epoch = 0

        for epoch in range(args.epoch):

            if stop_count == 80 or increase_count == 80:
                print('model convergent at epoch {}, early stopping'.format(epoch))
                break

            epoch_train_loss = 0
            epoch_train_acc = 0
            num_iterations = 0

            model.train()
            train_embed = []

            for x, y in train_loader:
                optimizer.zero_grad()
                # print("raw x.shape = ", x.shape)
                xb, num_patch = create_patch(xb=x.permute(0,2,1), patch_len=args.patch_len, stride=args.stride)
                # print("patch xb.shape = ", xb.shape)

                pred = model(xb)
                step_loss = loss(pred, y)

                # step_loss.backward(retain_graph=True)
                step_loss.backward()
                optimizer.step()

                epoch_train_loss += step_loss.item()
                epoch_train_acc += torch.sum(torch.argmax(pred.data, axis=1) == y) / len(y)

                num_iterations += 1

            epoch_train_loss /= num_steps
            epoch_train_acc /= num_steps
            # train_embed = np.concatenate(train_embed)

            model.eval()

            val_loss, val_accu = evaluate_gpt4ts(args, val_loader, model, loss)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                end_val_epoch = epoch
                test_loss, test_accuracy = evaluate_gpt4ts(args, test_loader, model, loss)

            if abs(last_loss - val_loss) <= 1e-4:
                stop_count += 1
            else:
                stop_count = 0

            if val_loss > last_loss:
                increase_count += 1
            else:
                increase_count = 0

            last_loss = val_loss

            if epoch % 50 == 0:
                print(
                    "epoch : {}, train loss: {} , train accuracy : {}, \ntest_accuracy : {}".format(
                        epoch, epoch_train_loss, epoch_train_acc, test_accuracy))

        test_accuracies.append(test_accuracy)
        end_val_epochs.append(end_val_epoch)
        t = time.time() - t
        train_time += t

        print('{} fold finish training'.format(i))

    test_accuracies = torch.Tensor(test_accuracies)

    print("Training end: mean_test_acc = ", round(torch.mean(test_accuracies).item(), 4),
          "traning time (seconds) = ",
          round(train_time, 4), ", seed = ", args.random_seed)

    test_accuracies = test_accuracies.cpu().numpy()

    save_cls_new_result(args, np.mean(test_accuracies), np.max(test_accuracies), np.min(test_accuracies),
                        np.std(test_accuracies), train_time)

    print('Done!')
