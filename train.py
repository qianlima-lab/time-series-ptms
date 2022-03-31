import numpy as np
import argparse

from utils import build_model,  set_seed, build_dataset, build_loss, evaluate, save_finetune_result, get_all_datasets
from data.dataloader import UCRDataset
from data.preprocessing import  normalize_per_series, fill_nan_value
from torch.utils.data import DataLoader
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn or dilated')
    parser.add_argument('--task', type=str, default='classification', help='classification or reconstruction')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default=None, help='dataset(in ucr)')
    parser.add_argument('--dataroot', type=str, default=None, help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=0,  help='number of class')


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
    parser.add_argument('--batch_size', type=int, default=128, help='(32, 128) larger batch size on the big dataset, ')
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch')
    parser.add_argument('--mode', type=str, default='pretrain', help='train mode, default pretrain')
    parser.add_argument('--save_dir', type=str, default='./result')
    parser.add_argument('--continue_training', type=int, default=0, help='continue training')

    # Decoder setup
    parser.add_argument('--decoder_backbone', type=str, default='rnn', help='backbone of the decoder')
    

    # classifier setup
    parser.add_argument('--classifier', type=str, default='nonlinear', help='type of classifier(linear or nonlinear)')
    parser.add_argument('--classifier_input', type=int, default=320, help='input dim of the classifiers')
    parser.add_argument('--classifier_embedding', type=int, default=128, help='embedding dim of the non linear classifier')

    # fintune setup
    parser.add_argument('--source_dataset', type=str, default=None, help='source dataset of the pretrained model')
    parser.add_argument('--direct_train')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args)



    sum_dataset, sum_target, num_classes = build_dataset(args)
    args.num_classes = num_classes

    model, classifier = build_model(args)
    model, classifier = model.to(device), classifier.to(device)
    loss = build_loss(args).to(device)
    classifier_init_state = classifier.state_dict()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params' : classifier.parameters()}], 
        lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)


    if args.mode == 'pretrain' and args.task == 'classification':
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

        if not os.path.exists(os.path.join(args.save_dir, args.dataset)):
            os.mkdir(os.path.join(args.save_dir, args.dataset))
        
        if args.continue_training != 0:
            model.load_state_dict(torch.load(os.path.join(args.save_dir, args.dataset, 'pretrain_weights.pt')))
            classifier.load_state_dict(torch.load(os.path.join(args.save_dir, args.dataset, 'classifier_weights.pt')))

        print('{} started pretrain'.format(args.dataset))
        

        sum_dataset = normalize_per_series(sum_dataset)
        train_set = UCRDataset(torch.from_numpy(sum_dataset).to(device), torch.from_numpy(sum_target).to(device).to(torch.int64))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0)
        
        
        last_loss = 100
        stop_count = 0
        increase_count = 0

        min_loss = torch.inf
        min_epoch = 0
        model_to_save = None

        num_steps = train_set.__len__() // args.batch_size
        for epoch in range(args.epoch-args.continue_training):
            
            if stop_count == 10 or increase_count == 10:
                print("model convergent at epoch {}, early stopping.".format(epoch))
                break
                
            model.train()
            classifier.train()
            epoch_loss = 0
            epoch_accu = 0
            for x, y in train_loader:
         
                optimizer.zero_grad()
                pred = model(x)

                pred = classifier(pred)
                
                step_loss = loss(pred, y)

                step_loss.backward()
                optimizer.step()

                epoch_loss += step_loss.item()
                epoch_accu += torch.sum(torch.argmax(pred.data, axis=1) == y) / len(y)
            
            epoch_loss /= num_steps
            if abs(epoch_loss - last_loss) <= 1e-4:
                stop_count += 1
            else:
                stop_count = 0

            if epoch_loss > last_loss:
                increase_count += 1
            else:
                increase_count = 0
                
            last_loss = epoch_loss
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                min_epoch = epoch
                model_to_save = model.state_dict()

            epoch_accu /= num_steps
            if epoch % 100 == 0:
                print("epoch : {}, loss : {}, accuracy : {}".format(epoch, epoch_loss, epoch_accu))
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.dataset, 'pretrain_weights.pt'))
                torch.save(classifier.state_dict(), os.path.join(args.save_dir, args.dataset, 'classifier_weights.pt'))
        
        print('{} finished pretrain, with min loss {} at epoch {}'.format(args.dataset, min_loss, min_epoch))
        torch.save(model_to_save, os.path.join(args.save_dir, args.dataset, 'pretrain_weights.pt'))


    if args.mode == 'finetune':
        print('start finetune on {}'.format(args.dataset))

        train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(sum_dataset, sum_target)

        losses = []
        accuracies = []
        for i, train_dataset in enumerate(train_datasets):
            model.load_state_dict(torch.load(os.path.join(args.save_dir, args.source_dataset, 'pretrain_weights.pt')))
            classifier.load_state_dict(classifier_init_state)
            print('{} fold start training and evaluate'.format(i))
            max_accuracy = 0

            train_target = train_targets[i]
            val_dataset = val_datasets[i]
            val_target = val_targets[i]

            test_dataset = test_datasets[i]
            test_target = test_targets[i]

            train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

            # TODO normalize per series

            test_dataset = normalize_per_series(test_dataset)
            train_dataset = normalize_per_series(train_dataset)
            val_dataset = normalize_per_series(val_dataset)

            train_set = UCRDataset(torch.from_numpy(train_dataset).to(device), torch.from_numpy(train_target).to(device).to(torch.int64))
            val_set = UCRDataset(torch.from_numpy(val_dataset).to(device), torch.from_numpy(val_target).to(device).to(torch.int64))
            test_set = UCRDataset(torch.from_numpy(test_dataset).to(device), torch.from_numpy(test_target).to(device).to(torch.int64))
            
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

            train_loss = []
            train_accuracy = []
            num_steps = args.epoch // args.batch_size

            last_loss = 100
            stop_count = 0
            increase_count = 0

            num_steps = train_set.__len__() // args.batch_size
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
                test_loss, test_accu = evaluate(test_loader, model, classifier, loss, device)

                if epoch % 100 == 0:
                    print("epoch : {}, train loss: {} , train accuracy : {}, \nval loss : {}, val accuracy : {}, \ntest loss : {}, test accuracy : {}".format(epoch, epoch_train_loss, epoch_train_acc, val_loss, val_accu, test_loss, test_accu))
                
                
                train_loss.append(val_loss)
                train_accuracy.append(test_accu)

                max_accuracy = max(max_accuracy, test_accu)

                if abs(last_loss-val_loss) <= 1e-4:
                    stop_count += 1
                else:
                    stop_count = 0 

                if val_loss > last_loss:
                    increase_count += 1
                else:
                    increase_count = 0

                last_loss = val_loss
            losses.append(train_loss)
            accuracies.append(max_accuracy)

            print('{} fold finish training'.format(i))

        accuracies = torch.Tensor(accuracies)
        save_finetune_result(args, torch.mean(accuracies), torch.std(accuracies))
        print('Done!')
    

    if args.mode == 'pretrain' and args.task == 'reconstruction':
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

        if not os.path.exists(os.path.join(args.save_dir, args.dataset)):
            os.mkdir(os.path.join(args.save_dir, args.dataset))
        print('start reconstruction on {}'.format(args.dataset))

        sum_dataset, sum_target, num_classes = build_dataset(args)
        args.num_classes = num_classes

        sum_dataset = normalize_per_series(sum_dataset)

        train_set = UCRDataset(torch.from_numpy(sum_dataset).to(device), torch.from_numpy(sum_target))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0)

        num_steps = train_set.__len__() // args.batch_size
        last_loss = 0
        stop_count = 0
        min_loss = 100
        increase_count = 0
        model_to_save = None

        for epoch in range(args.epoch):
            # remove early stopping
            '''
            if stop_count == 10 or increase_count == 10:
                print("model convergent at epoch {}, early stopping.".format(epoch))
                break
            '''

            model.train()
            classifier.train()
            epoch_loss = 0

            for i, (x, _) in enumerate(train_loader):
                # x -> (batch_size, sequence length)
                # x_features -> (batch_size, out_channels)
                # x_reversed -> (batch_size, sequence length), (xt, xt-1. ..., x1)
                optimizer.zero_grad()
                x_features = model(x)
                x_reversed = torch.fliplr(x)

                # x_reversed -> (batch_size, sequence length, 1)
                time_length = x.shape[1]

                out = x_reversed[:,:,0]
                
                hidden1 = x_features
                hidden2 = x_features
                hidden3 = x_features


                step_loss = 0
                for i in range (time_length):
                    hidden1, hidden2, hidden3, out = classifier(hidden1, hidden2, hidden3, out)
                    step_loss += loss(out, x_reversed[:, :,i])
                
                step_loss /= time_length
                epoch_loss += step_loss
                step_loss.backward()
                optimizer.step()

            epoch_loss /= num_steps

            if epoch % 100 == 0:
                print("epoch : {}, loss : {}".format(epoch, epoch_loss))

            if epoch_loss < min_loss:
                model_to_save = model.state_dict()
                min_loss = epoch_loss
            # early stopping judge
            if abs(epoch_loss-last_loss) < 1e-6:
                stop_count += 1
            else:
                stop_count = 0

            if epoch_loss > last_loss:
                increase_count += 1
            else:
                increase_count = 0
            
            last_loss = epoch_loss
        
        print('{} finished pretrain, with min loss {} '.format(args.dataset, min_loss))
        torch.save(model_to_save, os.path.join(args.save_dir, args.dataset, 'pretrain_weights.pt'))



