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

from gpt4ts.gpt4ts_utils import load_UEA, normalize_uea_set, UEADataset, save_cls_new_result, set_seed, fill_nan_value, get_all_datasets, build_loss, build_dataset

from timesnet.models.TimesNet import Model



def collate_fn(data, device, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X.to(device), targets.to(device), padding_masks.to(device)


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))



def evaluate_gpt4ts(args, val_loader, model, loss):
    val_loss = 0
    val_accu = 0

    sum_len = 0
    for data, target, padding_x_mask in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        with torch.no_grad():
            val_pred = model(data, padding_x_mask)
            val_loss += loss(val_pred, target).item()
            val_accu += torch.sum(torch.argmax(val_pred.data, axis=1) == target)
            sum_len += len(target)

    return val_loss / sum_len, val_accu / sum_len


if __name__ == '__main__':  ##
    parser = argparse.ArgumentParser()

    # UCR, TimesNet: ['HandOutlines', 'InlineSkate', 'StarLightCurves']
    # UEA, TimesNet: ['EigenWorms', 'LSST', 'StandWalkJump']

    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='StarLightCurves',
                        help='dataset(in ucr)')  # LSST Heartbeat Images  SelfRegulationSCP2
    # parser.add_argument('--dataroot', type=str, default='../UCRArchive_2018', help='path of UCR folder')
    # parser.add_argument('--dataroot', type=str, default='/dev_data/lz/time_series_pretrain/datasets/UCRArchive_2018',
    #                     help='path of UCR folder')
    # parser.add_argument('--dataroot', type=str, default='/SSD/lz/UCRArchive_2018', help='path of UCR folder')
    parser.add_argument('--dataroot', type=str, default='/SSD/lz/UCRArchive_2018', help='path of UEA folder')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--normalize_way', type=str, default='single', help='single or train_set')
    # parser.add_argument('--seq_len', type=int, default=46, help='seq_len')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # parser.add_argument('--patch_size', type=int, default=8, help='patch_size')
    # parser.add_argument('--stride', type=int, default=8, help='stride')

    parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')

    # Patch
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride between patch')

    # # RevIN
    # parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
    # # Model args
    # parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
    # parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
    # # parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
    # parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
    # parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
    # parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')

    # Semi training
    parser.add_argument('--labeled_ratio', type=float, default='0.1', help='0.1, 0.2, 0.4')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')   ###
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
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

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')


    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:1')

    parser.add_argument('--save_dir', type=str, default='/SSD/lz/time_series_label_noise/result')
    parser.add_argument('--save_csv_name', type=str, default='timesnet_ucr_supervised_0801_')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='type of classifier(linear or nonlinear)')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    sum_dataset, sum_target, num_classes = build_dataset(args)
    # sum_dataset, sum_target, num_classes = load_UEA(args.dataroot, args.dataset)
    # args.num_classes = num_classes
    # args.seq_len = sum_dataset.shape[1]

    sum_dataset = sum_dataset[:, :, np.newaxis]

    args.num_classes = num_classes
    args.seq_len = sum_dataset.shape[1]
    args.input_size = sum_dataset.shape[2]

    args.enc_in = sum_dataset.shape[2]

    # # get number of patches
    # num_patch = (max(args.seq_len, args.patch_len) - args.patch_len) // args.stride + 1
    # print('number of patches:', num_patch)

    while sum_dataset.shape[0] * 0.6 < args.batch_size:
        args.batch_size = args.batch_size // 2

    print("args.batch_size = ", args.batch_size, ", sum_dataset.shape = ", sum_dataset.shape)

    # get model
    model = Model(configs=args)


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

        train_set = UEADataset(torch.from_numpy(train_dataset).type(torch.FloatTensor).to(device).permute(0,2,1),
                               torch.from_numpy(train_target).type(torch.FloatTensor).to(device).to(torch.int64))
        val_set = UEADataset(torch.from_numpy(val_dataset).type(torch.FloatTensor).to(device).permute(0,2,1),
                             torch.from_numpy(val_target).type(torch.FloatTensor).to(device).to(torch.int64))
        test_set = UEADataset(torch.from_numpy(test_dataset).type(torch.FloatTensor).to(device).permute(0,2,1),
                              torch.from_numpy(test_target).type(torch.FloatTensor).to(device).to(torch.int64))

        # train_set = train_set.permute(0,2,1)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=True, collate_fn=lambda x: collate_fn(x, device, max_len=args.seq_len))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0, collate_fn=lambda x: collate_fn(x, device, max_len=args.seq_len))
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0, collate_fn=lambda x: collate_fn(x, device, max_len=args.seq_len))

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

            for x, y, padding_x_mask in train_loader:
                optimizer.zero_grad()
                # print("raw x.shape = ", x.shape)
                # xb, num_patch = create_patch(xb=x.permute(0,2,1), patch_len=args.patch_len, stride=args.stride)
                # print("x padding_x_mask.shape = ", x.shape, padding_x_mask.shape, padding_x_mask[0][:10])

                pred = model(x, padding_x_mask)
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
