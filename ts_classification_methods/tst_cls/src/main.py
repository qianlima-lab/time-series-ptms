"""
Written by George Zerveas
Modified by Ziyang Huang

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

from models.loss import NoFussCrossEntropyLoss, MaskedMSELoss
from dataprepare import *
from optimizers import get_optimizer
from models.loss import get_loss_module
from models.ts_transformer import model_factory
from datasets.datasplit import split_dataset
from datasets.data import data_factory, Normalizer
from utils import utils
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from options import Options
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
import json
import pickle
import time
import sys
import os
from copy import deepcopy
import logging

logging.basicConfig(
    format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")
# 3rd party packages
# Project modules


def main(config):

    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(
        os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(
        ' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    if config['multi_gpu']:
        device_ids = [0, 1]
    device = torch.device('cuda:{}'.format(config['gpu']) if (
        torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'],
                         n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features
    validation_method = 'StratifiedKFold'
    labels = my_data.labels_df.values.flatten()
    # Split dataset
    test_data = my_data
    # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    test_indices = None
    val_data = my_data
    val_indices = []

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0

    # 5 fold
    accus = []
    times = []
    end_epochs = []
    for i in range(5):
        fold_start_time = time.time()
        train_indices, val_indices, test_indices = split_dataset(data_indices=my_data.all_IDs,
                                                                 validation_method=validation_method,
                                                                 n_splits=1,
                                                                 validation_ratio=config['val_ratio'],
                                                                 # used only if test_indices not explicitly specified
                                                                 test_set_ratio=config['test_ratio'],
                                                                 test_indices=test_indices,
                                                                 random_seed=42,
                                                                 labels=labels, ith=i)
        logger.info('{} fold start training!'.format(i))
        logger.info("{} samples may be used for training".format(
            len(train_indices)))
        logger.info(
            "{} samples will be used for validation".format(len(val_indices)))
        logger.info("{} samples will be used for testing".format(
            len(test_indices)))

        # Create model
        logger.info("Creating model ...")
        if config['task'] == 'pretrain_and_finetune':
            model, classifier = model_factory(config, my_data, labels)
        else:
            model = model_factory(config, my_data)

        if config['global_reg']:
            weight_decay = config['l2_reg']
            output_reg = None
        else:
            weight_decay = 0
            output_reg = config['l2_reg']

        optim_class = get_optimizer(config['optimizer'])
        optimizer = optim_class(
            model.parameters(), lr=config['lr'], weight_decay=weight_decay)

        start_epoch = 0
        lr_step = 0  # current step index of `lr_step`
        lr = config['lr']  # current learning step
        # Load model and optimizer state

        if config['multi_gpu']:
            model = nn.DataParallel(model, device_ids)
            optimizer = nn.DataParallel(optimizer, device_ids)

            if config['task'] == 'pretrain_and_finetune':
                classifier = nn.DataParallel(classifier, device_ids)

        model.to(device)
        if config['task'] == 'pretrain_and_finetune':
            classifier.to(device)
        elif config['task'] == 'classification':
            if config['load_root'] is not None:
                model.load_state_dict(torch.load(os.path.join(
                    config['load_root'], config['source_dataset'], 'encoder_weights.pt'), device))
            classifier = model

        loss_module = MaskedMSELoss(reduction='none')
        classification_module = NoFussCrossEntropyLoss(reduction='none')

        if config['task'] == 'classification':
            loss_module = classification_module
        '''
        if config['multi_gpu']:
            loss_module = nn.DataParallel(loss_module, device_ids)
        '''
        # Initialize data generators
        if config['task'] == 'pretrain_and_finetune':
            dataset_class, collate_fn, runner_class, cls_data_class, cls_collate_fn, cls_runner_cls = pipeline_factory(
                config, device)
        else:
            dataset_class, collate_fn, runner_class = pipeline_factory(
                config, device)
            cls_data_class, cls_collate_fn, cls_runner_cls = dataset_class, collate_fn, runner_class
        train_df, val_df, test_df = fill_nan_and_normalize(
            my_data.feature_df.loc[train_indices], val_data.feature_df.loc[val_indices], test_data.feature_df.loc[test_indices], train_indices, val_indices, test_indices)

        test_dataset = dataset_class(
            test_data, test_indices, feature_df=test_df)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True,
                                 collate_fn=lambda x: cls_collate_fn(x, max_len=model.max_len))

        val_dataset = dataset_class(val_data, val_indices, feature_df=val_df)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

        # config['num_workers'],pin_memory=True

        train_dataset = dataset_class(
            my_data, train_indices, feature_df=train_df)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

        trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                               print_interval=config['print_interval'], console=config['console'])

        val_evaluator = runner_class(model, val_loader, device, loss_module,
                                     print_interval=config['print_interval'], console=config['console'])

        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                      print_interval=config['print_interval'], console=config['console'])

        tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

        # initialize with +inf or -inf depending on key metric
        best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16
        best_test = 1e16 if config['key_metric'] in NEG_METRICS else -1e16

        best_metrics = {}
        best_test_metrics = {}

        logger.info('Starting training...')
        stop_count = 0
        increase_count = 0
        last_loss = 1e16
        val_loss = 1e16
        best_epoch = 0
        for epoch in range(start_epoch + 1, config["epochs"] + 1):
            if stop_count == 50 or increase_count == 50:
                print('model convergent at epoch {}, early stopping'.format(epoch))
                break
            epoch_start_time = time.time()
            # dictionary of aggregate epoch metrics
            aggr_metrics_train = trainer.train_epoch(epoch)
            epoch_runtime = time.time() - epoch_start_time
            if epoch % 100 == 0:
                print("epoch : {}".format(epoch))

            if config['task'] == 'pretrain_and_finetune':
                aggr_metrics_val, best_metrics, best_value, condition = validate(val_evaluator, tensorboard_writer, config,
                                                                                 best_metrics, best_value, epoch)

                if condition or epoch == 1:
                    best_epoch = epoch
                    best_state_dict = deepcopy(model.state_dict())

            elif config['task'] == 'classification':
                aggr_metrics_val, best_metrics, best_value, condition = validate(val_evaluator, tensorboard_writer, config,
                                                                                 best_metrics, best_value, epoch)

                if condition or epoch == 1:
                    best_epoch = epoch
                    best_state_dict = deepcopy(model.state_dict())
                    _, best_test_metrics, best_test, _ = validate(test_evaluator, tensorboard_writer, config,
                                                                  best_test_metrics, best_test, epoch)

                val_loss = aggr_metrics_val['loss']
                if abs(last_loss - val_loss) <= 1e-4:
                    stop_count += 1
                else:
                    stop_count = 0

                if val_loss > last_loss:
                    increase_count += 1
                else:
                    increase_count = 0

                last_loss = val_loss

        # save encoder weights
        if config['task'] == 'classification_transfer':
            save_path = os.path.join(
                config['weights_save_path'], config['dataset'])
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for key, val in model.state_dict().items():
                if key.startswith('output_layer'):
                    state_dict.pop(key)
            torch.save(state_dict, os.path.join(
                save_path, 'encoder_weights.pt'))

        if config['task'] == 'pretrain_and_finetune':
            classifier_optimizer = optim_class(
                classifier.parameters(), lr=config['lr'], weight_decay=weight_decay)
            if config['multi_gpu']:
                classifier_optimizer = nn.DataParallel(
                    classifier_optimizer, device_ids)
            finetune_train_dataset = cls_data_class(
                my_data, train_indices, feature_df=train_df)
            finetune_train_loader = DataLoader(dataset=finetune_train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=8,
                                               collate_fn=lambda x: cls_collate_fn(x, max_len=classifier.max_len))
            test_dataset = cls_data_class(
                test_data, test_indices, feature_df=test_df)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=config['batch_size'],
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=True,
                                     collate_fn=lambda x: cls_collate_fn(x, max_len=classifier.max_len), drop_last=True)

            val_dataset = cls_data_class(
                val_data, val_indices, feature_df=val_df)
            val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    num_workers=8,
                                    pin_memory=True,
                                    collate_fn=lambda x: cls_collate_fn(x, max_len=classifier.max_len), drop_last=True)
            val_evaluator = cls_runner_cls(classifier, val_loader, device, classification_module,
                                           print_interval=config['print_interval'], console=config['console'])
            test_evaluator = cls_runner_cls(classifier, test_loader, device, classification_module,
                                            print_interval=config['print_interval'], console=config['console'])
            classifier_trainer = cls_runner_cls(classifier, finetune_train_loader, device, classification_module, classifier_optimizer, l2_reg=output_reg,
                                                print_interval=config['print_interval'], console=config['console'])
            state_dict = deepcopy(best_state_dict)

            for key, val in model.state_dict().items():
                if key.startswith('output_layer'):
                    state_dict.pop(key)

            #classifier.module.load_state_dict(state_dict, strict=False)

            for epoch in range(start_epoch + 1, 101):
                epoch_start_time = time.time()
                aggr_metrics_train = classifier_trainer.train_epoch(
                    epoch)  # dictionary of aggregate epoch metrics
                epoch_runtime = time.time() - epoch_start_time

                aggr_metrics_val, best_metrics, best_value, condition = validate(val_evaluator, tensorboard_writer, config,
                                                                                 best_metrics, best_value, epoch)
                if condition or epoch == 1:
                    _, best_test_metrics, best_test, _ = validate(test_evaluator, tensorboard_writer, config,
                                                                  best_test_metrics, best_test, epoch)

        logger.info('Best {} was {}. Other metrics: {}'.format(
            config['key_metric'], best_value, best_metrics))
        logger.info('{} fold training Done!'.format(i))

        fold_end_time = time.time()
        accus.append(best_test_metrics['accuracy'].cpu().numpy())
        times.append(fold_end_time-fold_start_time)
        end_epochs.append(best_epoch)
    # TODO 已经有了所有的metric，参照tsmutil将所有的插入表格并开始训练
    accus = np.array(accus)
    acc_mean = accus.mean()
    acc_std = accus.std()
    time_mean = np.array(times).mean()
    epoch_mean = np.array(end_epochs).mean()

    if config['task'] == 'pretrain_and_finetune':
        save_path = './tst_results.csv'
        if os.path.exists(save_path):
            result_form = pd.read_csv(save_path)
        else:
            result_form = pd.DataFrame(columns=['target', 'accuracy', 'std'])

        result_form = result_form.append(
            {'target': config['dataset'], 'accuracy': '%.4f' % acc_mean, 'std': '%.4f' % acc_std}, ignore_index=True)
        result_form = result_form.iloc[:, -3:]
        result_form.to_csv(save_path)

    elif config['task'] == 'classification':
        save_path = './non_linear_classification_tst_results.csv'
        if os.path.exists(save_path):
            result_form = pd.read_csv(save_path)
        else:
            result_form = pd.DataFrame(columns=[
                                       'dataset_name', 'test_accuracy', 'test_std', 'train_time', 'end_val_epoch', 'seeds'])

        result_form = result_form.append({'dataset_name': config['dataset'], 'test_accuracy': '%.4f' % acc_mean, 'test_std': '%.4f' % acc_std, 'train_time': '%.4f' % time_mean, 'end_val_epoch': '%.2f' % epoch_mean,
                                          'seeds': '%d' % 42}, ignore_index=True)
        result_form = result_form.iloc[:, -6:]
        result_form.to_csv(save_path)

    return best_value


if __name__ == '__main__':
    # set seed
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
