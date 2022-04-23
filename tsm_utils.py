import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim

from data.preprocessing import load_data, k_fold, transfer_labels
from model.loss import cross_entropy, reconstruction_loss
from model.tsm_model import FCN, DilatedConvolution, Classifier, NonLinearClassifier, RNNDecoder, FCNDecoder


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def build_model(args):
    if args.backbone == 'fcn':
        model = FCN(args.num_classes, args.input_size)
    elif args.backbone == 'dilated':
        model = DilatedConvolution(args.in_channels, args.embedding_channels,
                                   args.out_channels, args.depth, args.reduced_size, args.kernel_size, args.num_classes)

    if args.task == 'classification':
        if args.classifier == 'nonlinear':
            classifier = NonLinearClassifier(args.classifier_input, 128, args.num_classes)
        elif args.classifier == 'linear':
            classifier = Classifier(args.classifier_input, args.num_classes)

    elif args.task == 'reconstruction':
        if args.decoder_backbone == 'rnn':
            classifier = RNNDecoder()
        if args.decoder_backbone == 'fcn':
            classifier = FCNDecoder(num_classes=args.num_classes, seq_len=args.seq_len, input_size=args.input_size)

    return model, classifier


def build_dataset(args):
    sum_dataset, sum_target, num_classes = load_data(args.dataroot, args.dataset)

    sum_target = transfer_labels(sum_target)
    return sum_dataset, sum_target, num_classes


def build_loss(args):
    if args.loss == 'cross_entropy':
        return cross_entropy()
    elif args.loss == 'reconstruction':
        return reconstruction_loss()


def build_optimizer(args):
    if args.optimizer == 'adam':
        return torch.optim.Adam(lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay)


def evaluate(val_loader, model, classifier, loss, device):
    val_loss = 0
    val_accu = 0

    sum_len = 0
    for data, target in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        with torch.no_grad():
            val_pred = model(data)
            val_pred = classifier(val_pred)
            val_loss += loss(val_pred, target).item()

            val_accu += torch.sum(torch.argmax(val_pred.data, axis=1) == target)
            sum_len += len(target)

    return val_loss / sum_len, val_accu / sum_len


def save_finetune_result(args, accu, std):
    save_path = os.path.join(args.save_dir, args.source_dataset, 'finetune_result.csv')
    # save_path = os.path.join(args.save_dir, 'finetune_result.csv')
    accu = accu.cpu().numpy()
    std = std.cpu().numpy()
    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path)
    else:
        result_form = pd.DataFrame(columns=['dataset', 'accuracy', 'std'])

    result_form = result_form.append({'dataset': args.dataset, 'accuracy': '%.4f' % accu, 'std': '%.4f' % std},
                                     ignore_index=True)
    result_form = result_form.iloc[:, -3:]
    result_form.to_csv(save_path)


def save_cls_result(args, test_accu, test_std, train_time, end_val_epoch, seeds=42):
    save_path = os.path.join(args.save_dir, '', args.save_csv_name + 'cls_result.csv')
    accu = test_accu.cpu().numpy()
    std = test_std.cpu().numpy()
    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path, index_col=0)
    else:
        result_form = pd.DataFrame(
            columns=['dataset_name', 'test_accuracy', 'test_std', 'train_time', 'end_val_epoch', 'seeds'])

    result_form = result_form.append(
        {'dataset_name': args.dataset, 'test_accuracy': '%.4f' % accu, 'test_std': '%.4f' % std,
         'train_time': '%.4f' % train_time, 'end_val_epoch': '%.2f' % end_val_epoch,
         'seeds': '%d' % seeds}, ignore_index=True)

    result_form.to_csv(save_path, index=True, index_label="id")


def get_all_datasets(data, target):
    return k_fold(data, target)
