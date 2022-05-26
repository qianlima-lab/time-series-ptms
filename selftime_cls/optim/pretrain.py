# -*- coding: utf-8 -*-

import torch
import utils.transforms as transforms
from dataloader.ucr2018 import *
import torch.utils.data as data
from model.model_RelationalReasoning import *
from model.model_backbone import SimConv4

def pretrain_IntraSampleRel(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

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
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)

    if '2C' in opt.class_type:
        cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
        nb_class=2
    elif '3C' in opt.class_type:
        cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
        nb_class=3
    elif '4C' in opt.class_type:
        cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
        nb_class=4
    elif '5C' in opt.class_type:
        cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
        nb_class = 5
    elif '6C' in opt.class_type:
        cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
        nb_class = 6
    elif '7C' in opt.class_type:
        cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
        nb_class = 7
    elif '8C' in opt.class_type:
        cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
        nb_class = 8

    tensor_transform = transforms.ToTensor()

    backbone = SimConv4().cuda()
    model = RelationalReasoning_Intra(backbone, feature_size, nb_class).cuda()

    train_set = MultiUCR2018_Intra(data=x_train, targets=y_train, K=K,
                               transform=train_transform, transform_cut=cut_piece,
                               totensor_transform=tensor_transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max


def pretrain_InterSampleRel(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

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
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)
    train_transform = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    backbone = SimConv4().cuda()
    model = RelationalReasoning(backbone, feature_size).cuda()

    train_set = MultiUCR2018(data=x_train, targets=y_train, K=K, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max


def pretrain_SelfTime(x_train, y_train, opt, in_channel=1):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': jitter,
                       'cutout': cutout,
                       'scaling': scaling,
                       'magnitude_warp': magnitude_warp,
                       'time_warp': time_warp,
                       'window_slice': window_slice,
                       'window_warp': window_warp,
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = [transforms_list[name] for name in opt.aug_type]
    train_transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()

    if '2C' in opt.class_type:
        cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
        nb_class=2
    elif '3C' in opt.class_type:
        cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
        nb_class=3
    elif '4C' in opt.class_type:
        cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
        nb_class=4
    elif '5C' in opt.class_type:
        cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
        nb_class = 5
    elif '6C' in opt.class_type:
        cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
        nb_class = 6
    elif '7C' in opt.class_type:
        cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
        nb_class = 7
    elif '8C' in opt.class_type:
        cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
        nb_class = 8

    backbone = SimConv4(in_channel).cuda()
    model = RelationalReasoning_InterIntra(backbone, feature_size, nb_class).cuda()

    train_set = MultiUCR2018_InterIntra(data=x_train, targets=y_train, K=K,
                                        transform=train_transform, transform_cut=cut_piece,
                                        totensor_transform=tensor_transform)
    
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max, model.backbone.state_dict()



