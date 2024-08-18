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
from data.preprocessing import *
import os
import json
import math
import torch
import numpy
import pandas
import argparse
import pickle
import scikit_wrappers
import sklearn

import sys
sys.path.append('..')


def load_UCR_dataset(path, dataset):
    """
    Loads the UCR dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    train_file = os.path.join(path, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(path, dataset, dataset + "_TEST.tsv")
    train_df = pandas.read_csv(train_file, sep='\t', header=None)
    test_df = pandas.read_csv(test_file, sep='\t', header=None)
    train_array = numpy.array(train_df)
    test_array = numpy.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = numpy.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = numpy.expand_dims(train_array[:, 1:], 1).astype(numpy.float64)
    train_labels = numpy.vectorize(transform.get)(train_array[:, 0])
    test = numpy.expand_dims(test_array[:, 1:], 1).astype(numpy.float64)
    test_labels = numpy.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train, train_labels, test, test_labels
    # Post-publication note:
    # Using the testing set to normalize might bias the learned network,
    # but with a limited impact on the reported results on few datasets.
    # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
    mean = numpy.nanmean(numpy.concatenate([train, test]))
    var = numpy.nanvar(numpy.concatenate([train, test]))
    train = (train - mean) / math.sqrt(var)
    test = (test - mean) / math.sqrt(var)
    return train, train_labels, test, test_labels


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
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH',
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of hyperparameters to use; ' +
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
    '''
    train, train_labels, test, test_labels = load_UCR_dataset(
        args.path, args.dataset
    )
    '''
   # set seed
    set_seed(args)

    sum_dataset, sum_target, num_classes = load_data(args.path, args.dataset)
    '''
    sum_dataset = normalize_per_series(sum_dataset)
    sum_dataset = numpy.expand_dims(sum_dataset, 1).astype(numpy.float64)
    '''
    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = k_fold(
        sum_dataset, sum_target)
    accuracies = []
    for i, train_dataset in enumerate(train_datasets):
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

        train_dataset,  test_dataset = numpy.expand_dims(train_dataset, 1).astype(
            numpy.float64), numpy.expand_dims(test_dataset, 1).astype(numpy.float64)
        if not args.load and not args.fit_classifier:

            classifier = fit_hyperparameters(
                args.hyper, train_dataset, train_target, args.cuda, args.gpu
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
            '''
            classifier.save(
                os.path.join(args.save_path, args.dataset)
            )
            with open(
                os.path.join(
                    args.save_path, args.dataset + '_hyperparameters.json'
                ), 'w'
            ) as fp:
                json.dump(classifier.get_params(), fp)
            '''

        print("Test accuracy: " + str(classifier.score(test_dataset, test_target)))
        accuracies.append(classifier.score(test_dataset, test_target))
    accuracies = numpy.array(accuracies)

    if os.path.exists('./tloss_result.csv'):
        result_form = pd.read_csv('./tloss_result.csv')
    else:
        result_form = pd.DataFrame(columns=['target', 'accuracy', 'std'])

    result_form = result_form.append({'target': args.dataset, 'accuracy': '%.4f' % numpy.mean(
        accuracies), 'std': '%.4f' % numpy.std(accuracies)}, ignore_index=True)
    result_form = result_form.iloc[:, -3:]
    result_form.to_csv('./tloss_result.csv')
