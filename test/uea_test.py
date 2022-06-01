import numpy as np
import torch

from data.preprocessing import fill_nan_value, normalize_uea_set
from data.preprocessing import load_UEA

uea_all = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
           'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing',
           'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting',
           'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
           'NATOPS', 'PenDigits', 'PEMS-SF', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
           'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

uea_all = ['FaceDetection']

dataroot = '/SSD/lz/Multivariate2018_arff'
i = 0
for dataset_name in uea_all:

    sum_dataset, sum_target, num_classes = load_UEA(dataroot,
                                                    dataset_name)  ## (num_size, series_length, num_dimensions)

    series_length = []
    for t_data in sum_dataset:
        series_length, num_dimensions = t_data.shape
        print(series_length, num_dimensions)
    new_torch_sum = torch.tensor(sum_dataset).permute(0, 2, 1)
    print("i = ", i, ", dataset_name = ", dataset_name, ", shape = ", sum_dataset.shape, new_torch_sum.shape,
          num_classes)

    if np.isnan(sum_dataset).any():
        print("There has nan!!!!!!!!!!")
        sum_dataset, _, _ = fill_nan_value(sum_dataset, sum_dataset, sum_dataset)
        sum_dataset = normalize_uea_set(sum_dataset)
        if np.isnan(sum_dataset).any():
            print("Still has nan!!!")
        else:
            print("Mean imputation success!!!")
    i += 1
