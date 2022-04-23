import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataloader import UEADataset
from data.preprocessing import fill_nan_value, normalize_train_val_test, load_UEA, \
    normalize_uea_set
from tsm_utils import build_model, set_seed, build_loss, evaluate, get_all_datasets, save_cls_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn or dilated')
    parser.add_argument('--task', type=str, default='classification', help='classification or reconstruction')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='BasicMotions', help='dataset(in ucr)')
    parser.add_argument('--dataroot', type=str, default='/SSD/lz/Multivariate2018_arff', help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--normalize_way', type=str, default='single', help='single or train_set')
    parser.add_argument('--seq_len', type=int, default=46, help='seq_len')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # Dilated Convolution setup
    parser.add_argument('--depth', type=int, default=3, help='depth of the dilated conv model')
    parser.add_argument('--in_channels', type=int, default=1, help='input data channel')
    parser.add_argument('--embedding_channels', type=int, default=40, help='mid layer channel')
    parser.add_argument('--reduced_size', type=int, default=160, help='number of channels after Global max Pool')
    parser.add_argument('--out_channels', type=int, default=320, help='number of channels after linear layer')
    parser.add_argument('--kernel_size', type=int, default=3, help='convolution kernel size')

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='(16, 128) larger batch size on the big dataset, ')
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch')
    parser.add_argument('--mode', type=str, default='directly_cls', help='train mode, default pretrain')
    parser.add_argument('--save_dir', type=str, default='/SSD/lz/time_tsm/result_tsm')
    parser.add_argument('--save_csv_name', type=str, default='ex1_test_uea_0423_')
    parser.add_argument('--continue_training', type=int, default=0, help='continue training')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # Decoder setup
    parser.add_argument('--decoder_backbone', type=str, default='rnn', help='backbone of the decoder (rnn or fcn)')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='nonlinear', help='type of classifier(linear or nonlinear)')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')
    parser.add_argument('--classifier_embedding', type=int, default=128,
                        help='embedding dim of the non linear classifier')

    # fintune setup
    parser.add_argument('--source_dataset', type=str, default=None, help='source dataset of the pretrained model')
    parser.add_argument('--transfer_strategy', type=str, default='classification',
                        help='classification or reconstruction')
    # parser.add_argument('--direct_train')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    # sum_dataset, sum_target, num_classes = build_dataset(args)
    sum_dataset, sum_target, num_classes = load_UEA(args.dataroot, args.dataset)
    args.num_classes = num_classes
    args.seq_len = sum_dataset.shape[1]
    args.input_size = sum_dataset.shape[2]
    # print("test: sum_dataset.shape = ", sum_dataset.shape)
    if sum_dataset.shape[0] * 0.6 < args.batch_size:
        args.batch_size = 16

    model, classifier = build_model(args)
    model, classifier = model.to(device), classifier.to(device)
    loss = build_loss(args).to(device)
    model_init_state = model.state_dict()
    classifier_init_state = classifier.state_dict()
    # print("model = ", model)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.mode == 'directly_cls':
        print('start finetune on {}'.format(args.dataset))

        train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
            sum_dataset, sum_target)

        losses = []
        test_accuracies = []
        train_time = 0.0
        end_val_epochs = []
        for i, train_dataset in enumerate(train_datasets):
            t = time.time()
            model.load_state_dict(model_init_state)
            classifier.load_state_dict(classifier_init_state)
            print('{} fold start training and evaluate'.format(i))

            train_target = train_targets[i]
            val_dataset = val_datasets[i]
            val_target = val_targets[i]

            test_dataset = test_datasets[i]
            test_target = test_targets[i]

            train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

            if args.normalize_way == 'single':
                train_dataset = normalize_uea_set(train_dataset)
                val_dataset = normalize_uea_set(val_dataset)
                test_dataset = normalize_uea_set(test_dataset)
            else:
                train_dataset, val_dataset, test_dataset = normalize_train_val_test(train_dataset, val_dataset,
                                                                                    test_dataset)

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

            test_accuracy = 0
            min_val_loss = float('inf')
            end_val_epoch = 0

            num_steps = train_set.__len__() // args.batch_size
            # print("test, args.batch_size = ", args.batch_size, ", num_steps = ", num_steps)
            for epoch in range(args.epoch):
                # early stopping in finetune
                if stop_count == 50 or increase_count == 50:
                    print('model convergent at epoch {}, early stopping'.format(epoch))
                    break

                epoch_train_loss = 0
                epoch_train_acc = 0
                model.train()
                classifier.train()
                for x, y in train_loader:
                    # print("type 2 = ", type(x))
                    optimizer.zero_grad()
                    pred = model(x)
                    pred = classifier(pred)

                    step_loss = loss(pred, y)
                    step_loss.backward()
                    optimizer.step()

                    epoch_train_loss += step_loss.item()
                    epoch_train_acc += torch.sum(torch.argmax(pred.data, axis=1) == y) / len(y)

                epoch_train_loss /= num_steps
                epoch_train_acc /= num_steps

                model.eval()
                classifier.eval()
                val_loss, val_accu = evaluate(val_loader, model, classifier, loss, device)
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    end_val_epoch = epoch
                    test_loss, test_accuracy = evaluate(test_loader, model, classifier, loss, device)

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
            test_accuracies.append(test_accuracy)
            end_val_epochs.append(end_val_epoch)
            t = time.time() - t
            train_time += t

            print('{} fold finish training'.format(i))

        test_accuracies = torch.Tensor(test_accuracies)
        end_val_epochs = np.array(end_val_epochs)
        save_cls_result(args, test_accu=torch.mean(test_accuracies), test_std=torch.std(test_accuracies),
                        train_time=train_time / 5, end_val_epoch=np.mean(end_val_epochs), seeds=args.random_seed)
        print('Done!')
