# -*- coding: utf-8 -*-

import torch
from optim.pytorchtools import EarlyStopping
import torch.nn as nn


class RelationalReasoning(torch.nn.Module):

  def __init__(self, backbone, feature_size=64):
    super(RelationalReasoning, self).__init__()
    self.backbone = backbone
    self.relation_head = torch.nn.Sequential(
                             torch.nn.Linear(feature_size*2, 256),
                             torch.nn.BatchNorm1d(256),
                             torch.nn.LeakyReLU(),
                             torch.nn.Linear(256, 1))

  def aggregate(self, features, K):
    relation_pairs_list = list()
    targets_list = list()
    size = int(features.shape[0] / K)
    shifts_counter=1
    for index_1 in range(0, size*K, size):
      for index_2 in range(index_1+size, size*K, size):
        # Using the 'cat' aggregation function by default
        pos1 = features[index_1:index_1 + size]
        pos2 = features[index_2:index_2+size]
        pos_pair = torch.cat([pos1,
                              pos2], 1)  # (batch_size, fz*2)

        # Shuffle without collisions by rolling the mini-batch (negatives)
        neg1 = torch.roll(features[index_2:index_2 + size],
                          shifts=shifts_counter, dims=0)
        neg_pair1 = torch.cat([pos1, neg1], 1) # (batch_size, fz*2)

        relation_pairs_list.append(pos_pair)
        relation_pairs_list.append(neg_pair1)

        targets_list.append(torch.ones(size, dtype=torch.float32).cuda())
        targets_list.append(torch.zeros(size, dtype=torch.float32).cuda())

        shifts_counter+=1
        if(shifts_counter>=size):
            shifts_counter=1 # avoid identity pairs
    relation_pairs = torch.cat(relation_pairs_list, 0).cuda()  # K(K-1) * (batch_size, fz*2)
    targets = torch.cat(targets_list, 0).cuda()
    return relation_pairs, targets

  def train(self, tot_epochs, train_loader, opt):
    patience = opt.patience
    early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    optimizer = torch.optim.Adam([
                  {'params': self.backbone.parameters()},
                  {'params': self.relation_head.parameters()}], lr=opt.learning_rate)
    BCE = torch.nn.BCEWithLogitsLoss()
    self.backbone.train()
    self.relation_head.train()
    epoch_max = 0
    acc_max=0
    for epoch in range(tot_epochs):

      acc_epoch=0
      loss_epoch=0
      # the real target is discarded (unsupervised)
      for i, (data_augmented, _) in enumerate(train_loader):
        K = len(data_augmented) # tot augmentations
        x = torch.cat(data_augmented, 0).cuda()

        optimizer.zero_grad()
        # forward pass (backbone)
        features = self.backbone(x)
        # aggregation function
        relation_pairs, targets = self.aggregate(features, K)

        # forward pass (relation head)
        score = self.relation_head(relation_pairs).squeeze()
        # cross-entropy loss and backward
        loss = BCE(score, targets)
        loss.backward()
        optimizer.step()
        # estimate the accuracy
        predicted = torch.round(torch.sigmoid(score))
        correct = predicted.eq(targets.view_as(predicted)).sum()
        accuracy = (100.0 * correct / float(len(targets)))
        acc_epoch += accuracy.item()
        loss_epoch += loss.item()

      acc_epoch /= len(train_loader)
      loss_epoch /= len(train_loader)

      if acc_epoch>acc_max:
          acc_max = acc_epoch
          epoch_max = epoch

      early_stopping(acc_epoch, self.backbone)
      if early_stopping.early_stop:
          print("Early stopping")
          break

      if (epoch+1)%opt.save_freq==0:
        print("[INFO] save backbone at epoch {}!".format(epoch))
        torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

      print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, Max ACC.= {:.1f}%, Max Epoch={}' \
            .format(epoch + 1, opt.model_name, opt.dataset_name,
                    loss_epoch, acc_epoch, acc_max, epoch_max))
    return acc_max, epoch_max


class RelationalReasoning_Intra(torch.nn.Module):

  def __init__(self, backbone, feature_size=64, nb_class=3):
    super(RelationalReasoning_Intra, self).__init__()
    self.backbone = backbone

    self.cls_head = torch.nn.Sequential(
        torch.nn.Linear(feature_size*2, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, nb_class),
        torch.nn.Softmax(),
    )

  def run_test(self, predict, labels):
      correct = 0
      pred = predict.data.max(1)[1]
      correct = pred.eq(labels.data).cpu().sum()
      return correct, len(labels.data)

  def train(self, tot_epochs, train_loader, opt):
    patience = opt.patience
    early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    optimizer = torch.optim.Adam([
                  {'params': self.backbone.parameters()},
        {'params': self.cls_head.parameters()},
    ], lr=opt.learning_rate)
    c_criterion = nn.CrossEntropyLoss()

    self.backbone.train()
    self.cls_head.train()
    epoch_max = 0
    acc_max=0
    for epoch in range(tot_epochs):

      acc_epoch=0
      acc_epoch_cls=0
      loss_epoch=0
      # the real target is discarded (unsupervised)
      for i, (data_augmented0, data_augmented1, data_label, _) in enumerate(train_loader):
        K = len(data_augmented0) # tot augmentations
        x_cut0 = torch.cat(data_augmented0, 0).cuda()
        x_cut1 = torch.cat(data_augmented1, 0).cuda()
        c_label = torch.cat(data_label, 0).cuda()

        optimizer.zero_grad()
        # forward pass (backbone)
        features_cut0 = self.backbone(x_cut0)
        features_cut1 = self.backbone(x_cut1)
        features_cls = torch.cat([features_cut0, features_cut1], 1)

        # score_intra = self.relation_head(relation_pairs_intra).squeeze()
        c_output = self.cls_head(features_cls)
        correct_cls, length_cls = self.run_test(c_output, c_label)

        loss_c = c_criterion(c_output, c_label)
        loss=loss_c

        loss.backward()
        optimizer.step()
        # estimate the accuracy
        loss_epoch += loss.item()

        accuracy_cls = 100. * correct_cls / length_cls
        acc_epoch_cls += accuracy_cls.item()

      acc_epoch_cls /= len(train_loader)
      loss_epoch /= len(train_loader)

      if acc_epoch_cls>acc_max:
          acc_max = acc_epoch_cls
          epoch_max = epoch

      early_stopping(acc_epoch_cls, self.backbone)
      if early_stopping.early_stop:
          print("Early stopping")
          break

      if (epoch+1)%opt.save_freq==0:
        print("[INFO] save backbone at epoch {}!".format(epoch))
        torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

      print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, '
            'Max ACC.= {:.1f}%, Max Epoch={}' \
            .format(epoch + 1, opt.model_name, opt.dataset_name,
                    loss_epoch, acc_epoch,acc_epoch_cls, acc_max, epoch_max))
    return acc_max, epoch_max


class RelationalReasoning_InterIntra(torch.nn.Module):
  def __init__(self, backbone, feature_size=64, nb_class=3):
    super(RelationalReasoning_InterIntra, self).__init__()
    self.backbone = backbone

    self.relation_head = torch.nn.Sequential(
                             torch.nn.Linear(feature_size*2, 256),
                             torch.nn.BatchNorm1d(256),
                             torch.nn.LeakyReLU(),
                             torch.nn.Linear(256, 1))
    self.cls_head = torch.nn.Sequential(
        torch.nn.Linear(feature_size*2, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, nb_class),
        torch.nn.Softmax(),
    )
    # self.softmax = nn.Softmax()

  def aggregate(self, features, K):
    relation_pairs_list = list()
    targets_list = list()
    size = int(features.shape[0] / K)
    shifts_counter=1
    for index_1 in range(0, size*K, size):
      for index_2 in range(index_1+size, size*K, size):

        # Using the 'cat' aggregation function by default
        pos1 = features[index_1:index_1 + size]
        pos2 = features[index_2:index_2+size]
        pos_pair = torch.cat([pos1,
                              pos2], 1)  # (batch_size, fz*2)

        # Shuffle without collisions by rolling the mini-batch (negatives)
        neg1 = torch.roll(features[index_2:index_2 + size],
                          shifts=shifts_counter, dims=0)
        neg_pair1 = torch.cat([pos1, neg1], 1) # (batch_size, fz*2)

        relation_pairs_list.append(pos_pair)
        relation_pairs_list.append(neg_pair1)

        targets_list.append(torch.ones(size, dtype=torch.float32).cuda())
        targets_list.append(torch.zeros(size, dtype=torch.float32).cuda())

        shifts_counter+=1
        if(shifts_counter>=size):
            shifts_counter=1 # avoid identity pairs
    relation_pairs = torch.cat(relation_pairs_list, 0).cuda()  # K(K-1) * (batch_size, fz*2)
    targets = torch.cat(targets_list, 0).cuda()
    return relation_pairs, targets

  def run_test(self, predict, labels):
      correct = 0
      pred = predict.data.max(1)[1]
      correct = pred.eq(labels.data).cpu().sum()
      return correct, len(labels.data)

  def train(self, tot_epochs, train_loader, opt):
    patience = opt.patience
    early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    optimizer = torch.optim.Adam([
                  {'params': self.backbone.parameters()},
                  {'params': self.relation_head.parameters()},
        {'params': self.cls_head.parameters()},
    ], lr=opt.learning_rate)
    BCE = torch.nn.BCEWithLogitsLoss()
    c_criterion = nn.CrossEntropyLoss()

    self.backbone.train()
    self.relation_head.train()
    self.cls_head.train()
    epoch_max = 0
    acc_max=0
    for epoch in range(tot_epochs):

      acc_epoch=0
      acc_epoch_cls=0
      loss_epoch=0
      # the real target is discarded (unsupervised)
      for i, (data, data_augmented0, data_augmented1, data_label, _) in enumerate(train_loader):
        K = len(data) # tot augmentations
        x = torch.cat(data, 0)
        x_cut0 = torch.cat(data_augmented0, 0)
        x_cut1 = torch.cat(data_augmented1, 0)
        c_label = torch.cat(data_label, 0)
      
        optimizer.zero_grad()
        # forward pass (backbone)
        features = self.backbone(x)
        features_cut0 = self.backbone(x_cut0)
        features_cut1 = self.backbone(x_cut1)

        features_cls = torch.cat([features_cut0, features_cut1], 1)

        # aggregation function
        relation_pairs, targets = self.aggregate(features, K)
        # relation_pairs_intra, targets_intra = self.aggregate_intra(features_cut0, features_cut1, K)

        # forward pass (relation head)
        score = self.relation_head(relation_pairs).squeeze()
        c_output = self.cls_head(features_cls)
        correct_cls, length_cls = self.run_test(c_output, c_label)

        # cross-entropy loss and backward
        loss = BCE(score, targets)
        loss_c = c_criterion(c_output, c_label)
        loss+=loss_c

        loss.backward()
        optimizer.step()
        # estimate the accuracy
        predicted = torch.round(torch.sigmoid(score))
        correct = predicted.eq(targets.view_as(predicted)).sum()
        accuracy = (100.0 * correct / float(len(targets)))
        acc_epoch += accuracy.item()
        loss_epoch += loss.item()

        accuracy_cls = 100. * correct_cls / length_cls
        acc_epoch_cls += accuracy_cls.item()

      acc_epoch /= len(train_loader)
      acc_epoch_cls /= len(train_loader)
      loss_epoch /= len(train_loader)

      if (acc_epoch+acc_epoch_cls)>acc_max:
          acc_max = (acc_epoch+acc_epoch_cls)
          epoch_max = epoch

      early_stopping((acc_epoch+acc_epoch_cls), self.backbone)
      if early_stopping.early_stop:
          print("Early stopping")
          break

      if (epoch+1)%opt.save_freq==0:
        print("[INFO] save backbone at epoch {}!".format(epoch))
        torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

      print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, '
            'Max ACC.= {:.1f}%, Max Epoch={}' \
            .format(epoch + 1, opt.model_name, opt.dataset_name,
                    loss_epoch, acc_epoch,acc_epoch_cls, acc_max, epoch_max))
    return acc_max, epoch_max




