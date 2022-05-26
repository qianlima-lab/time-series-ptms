# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from tsm_utils import set_seed
from data import load_UEA, UEADataset, k_fold, fill_nan_value, normalize_per_series
import pandas as pd
import os
import json
import math
import torch
import numpy
import argparse
'''
import weka.core.jvm
import weka.core.converters
'''
import time
import scikit_wrappers
import sys
sys.path.append('..')
sys.path.remove('..')


def fit_hyperparameters(file, train, train_labels, cuda, gpu,
                        save_memory=False):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = scikit_wrappers.CausalCNNEncoderClassifier()

    # Loads a given set of hyperparameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = numpy.shape(train)[1]
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)
    return classifier.fit(
        train, train_labels, save_memory=save_memory, verbose=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=False,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of hyperparameters to use ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')

    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False
    set_seed(args)
    sum_dataset, sum_target, num_classes = load_UEA(args.path, args.dataset)

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = k_fold(
        sum_dataset, sum_target)
    accuracies = []
    times = []
    for i, train_dataset in enumerate(train_datasets):
        start = time.time()
        print('{} fold start training!'.format(i+1))
        train_target = train_targets[i]

        val_target = val_targets[i]
        val_dataset = val_datasets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        train_dataset, _, test_dataset = fill_nan_value(
            train_dataset, val_dataset, test_dataset)
        train_dataset, test_dataset = normalize_per_series(
            train_dataset), normalize_per_series(test_dataset)

        train_dataset = numpy.concatenate((train_dataset, val_dataset))
        train_target = numpy.concatenate((train_target, val_target))

        if not args.load and not args.fit_classifier:
            classifier = fit_hyperparameters(
                args.hyper, train_dataset, train_target, args.cuda, args.gpu,
                save_memory=True
            )
        else:
            classifier = scikit_wrappers.CausalCNNEncoderClassifier()
            hf = open(
                os.path.join(
                    args.save_path, args.dataset + '_hyperparameters.json'
                ), 'r'
            )
            hp_dict = json.load(hf)
            hf.close()
            hp_dict['cuda'] = args.cuda
            hp_dict['gpu'] = args.gpu
            classifier.set_params(**hp_dict)
            classifier.load(os.path.join(args.save_path, args.dataset))

        if not args.load:
            if args.fit_classifier:
                classifier.fit_classifier(
                    classifier.encode(train_dataset), train_target)

        accu = classifier.score(test_dataset, test_target)
        print("Test accuracy: " + str(accu))
        end = time.time()
        times.append(end-start)
        accuracies.append(accu)

    accuracies = numpy.array(accuracies)
    times = numpy.array(times)

    if os.path.exists('./tloss_uea.csv'):
        result_form = pd.read_csv('./tloss_uea.csv')
    else:
        result_form = pd.DataFrame(
            columns=['target', 'accuracy', 'std', 'times'])

    result_form = result_form.append({'target': args.dataset, 'accuracy': '%.4f' % numpy.mean(
        accuracies), 'std': '%.4f' % numpy.std(accuracies), 'times': '%.4f' % numpy.mean(times)}, ignore_index=True)
    result_form = result_form.iloc[:, -4:]
    result_form.to_csv('./tloss_uea.csv')
