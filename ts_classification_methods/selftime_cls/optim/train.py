# -*- coding: utf-8 -*-


import torch
import utils.transforms as transforms
from dataloader.ucr2018 import UCR2018
import torch.utils.data as data
from optim.pytorchtools import EarlyStopping
from model.model_backbone import SimConv4
import utils.transforms as transforms_ts


def supervised_train(x_train, y_train, x_val, y_val, x_test, y_test, nb_class, opt):
    # construct data loader
    # Those are the transformations used in the paper
    prob = 0.2  # Transform Probability
    cutout = transforms_ts.Cutout(sigma=0.1, p=prob)
    jitter = transforms_ts.Jitter(sigma=0.2, p=prob)  # CIFAR10
    scaling = transforms_ts.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms_ts.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms_ts.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms_ts.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms_ts.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms_ts.Compose(transforms_targets + [transforms_ts.ToTensor()])
    transform_lineval = transforms.Compose([transforms.ToTensor()])

    train_set_lineval = UCR2018(data=x_train, targets=y_train, transform=train_transform)
    val_set_lineval = UCR2018(data=x_val, targets=y_val, transform=transform_lineval)
    test_set_lineval = UCR2018(data=x_test, targets=y_test, transform=transform_lineval)

    train_loader_lineval = torch.utils.data.DataLoader(train_set_lineval, batch_size=128, shuffle=True)
    val_loader_lineval = torch.utils.data.DataLoader(val_set_lineval, batch_size=128, shuffle=False)
    test_loader_lineval = torch.utils.data.DataLoader(test_set_lineval, batch_size=128, shuffle=False)

    # loading the saved backbone
    backbone_lineval = SimConv4().cuda()  # defining a raw backbone model

    # 64 are the number of output features in the backbone, and 10 the number of classes
    linear_layer = torch.nn.Linear(opt.feature_size, nb_class).cuda()
    optimizer = torch.optim.Adam([{'params': backbone_lineval.parameters()},
                  {'params': linear_layer.parameters()}], lr=opt.learning_rate)

    CE = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(opt.patience_test, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    torch.save(backbone_lineval.state_dict(), '{}/backbone_init.tar'.format(opt.ckpt_dir))

    best_acc = 0
    best_epoch = 0

    print('Supervised Train')
    for epoch in range(opt.epochs_test):
        backbone_lineval.train()
        linear_layer.train()

        acc_trains = list()
        for i, (data, target) in enumerate(train_loader_lineval):
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()

            output = backbone_lineval(data)
            output = linear_layer(output)
            loss = CE(output, target)
            loss.backward()
            optimizer.step()
            # estimate the accuracy
            prediction = output.argmax(-1)
            correct = prediction.eq(target.view_as(prediction)).sum()
            accuracy = (100.0 * correct / len(target))
            acc_trains.append(accuracy.item())

        print('[Train-{}][{}] loss: {:.5f}; \t Acc: {:.2f}%' \
              .format(epoch + 1, opt.model_name, loss.item(), sum(acc_trains) / len(acc_trains)))

        acc_vals = list()
        acc_tests = list()
        backbone_lineval.eval()
        linear_layer.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader_lineval):
                data = data.cuda()
                target = target.cuda()

                output = backbone_lineval(data).detach()
                output = linear_layer(output)
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_vals.append(accuracy.item())

            val_acc = sum(acc_vals) / len(acc_vals)
            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                for i, (data, target) in enumerate(test_loader_lineval):
                    data = data.cuda()
                    target = target.cuda()

                    output = backbone_lineval(data).detach()
                    output = linear_layer(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_tests.append(accuracy.item())

                test_acc = sum(acc_tests) / len(acc_tests)

        print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
            epoch, val_acc, test_acc, best_epoch))
        early_stopping(val_acc, backbone_lineval)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(backbone_lineval.state_dict(), '{}/backbone_last.tar'.format(opt.ckpt_dir))

    return test_acc, best_epoch




