source_datasets = ['Crop', 'ElectricDevices', 'StarLightCurves', 'Wafer', 'ECG5000', 'TwoPatterns', 'FordA',
                   'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
                   'FordB', 'ChlorineConcentration', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2']

target_min_datasets = ['BirdChicken', 'BeetleFly', 'Coffee', 'OliveOil', 'Beef', 'Rock', 'ShakeGestureWiimoteZ',
                       'PickupGestureWiimoteZ', 'Wine', 'FaceFour', 'Meat', 'Car', 'Lightning2', 'Herring',
                       'Lightning7']

target_med_datasets = ['Earthquakes', 'Haptics', 'Computers', 'DistalPhalanxTW', 'DistalPhalanxOutlineAgeGroup',
                       'MiddlePhalanxTW', 'MiddlePhalanxOutlineAgeGroup',
                       'SyntheticControl', 'ProximalPhalanxTW', 'ProximalPhalanxOutlineAgeGroup',
                       'SonyAIBORobotSurface1', 'InlineSkate', 'EOGVerticalSignal', 'EOGHorizontalSignal',
                       'SmallKitchenAppliances']

target_max_datasets = ['MoteStrain', 'HandOutlines', 'CinCECGTorso', 'Phoneme', 'InsectWingbeatSound', 'FacesUCR',
                       'FaceAll',
                       'Mallat', 'MixedShapesSmallTrain', 'PhalangesOutlinesCorrect', 'FreezerSmallTrain',
                       'MixedShapesRegularTrain', 'FreezerRegularTrain', 'Yoga', 'MelbournePedestrian']

target_datasets = target_min_datasets + target_med_datasets + target_max_datasets
print(target_datasets)
print(len(source_datasets), len(target_datasets))

i = 0
for dataset in source_datasets:  ## cls pretrain
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1
    '''
    python train.py --backbone fcn --task classification --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --mode pretrain --epoch 20  --loss cross_entropy --cuda cuda:1;
    '''
    with open('/SSD/lz/time_tsm/scripts/transfer_pretrain_finetune.sh', 'a') as f:
        f.write(
            'python train.py --backbone fcn --task classification --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
            '--dataroot /SSD/lz/UCRArchive_2018 '
            '--dataset ' + dataset
            + ' --mode pretrain --epoch 2000 --classifier linear' +
            ' --loss cross_entropy --cuda cuda:1' + ';\n')

i = 0
for dataset in source_datasets:  ## rec fcn pretrain
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1
    '''
      python train.py --backbone fcn --task reconstruction --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --mode pretrain --epoch 20  --loss reconstruction --decoder_backbone fcn --cuda cuda:1;
    '''
    with open('/SSD/lz/time_tsm/scripts/transfer_pretrain_finetune.sh', 'a') as f:
        f.write(
            'python train.py --backbone fcn --task reconstruction --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
            '--dataroot /SSD/lz/UCRArchive_2018 '
            '--dataset ' + dataset
            + ' --mode pretrain --epoch 2000 --classifier linear' +
            ' --loss reconstruction --decoder_backbone fcn --cuda cuda:1' + ';\n')

i = 0
for dataset in source_datasets:  ## rec rnn pretrain
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1

    '''
     python train.py --backbone fcn --task reconstruction --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --mode pretrain --epoch 20  --loss reconstruction --decoder_backbone rnn --cuda cuda:1;
    '''
    with open('/SSD/lz/time_tsm/scripts/transfer_pretrain_finetune.sh', 'a') as f:
        f.write(
            'python train.py --backbone fcn --task reconstruction --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
            '--dataroot /SSD/lz/UCRArchive_2018 '
            '--dataset ' + dataset
            + ' --mode pretrain --epoch 2000 --classifier linear' +
            ' --loss reconstruction --decoder_backbone rnn --cuda cuda:1' + ';\n')

i = 0
for source_dataset in source_datasets:  ## cls finetune
    print("i = ", i, "dataset_name = ", source_dataset)
    i = i + 1
    for target_dataset in target_datasets:
        ### finetune cls
        '''
         python train.py --backbone fcn --task classification --classifier linear --classifier_input 128 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --mode finetune --epoch 20  --loss cross_entropy --source_dataset Coffee --transfer_strategy classification --cuda cuda:1 --save_csv_name test_fcn_nonlin_single_norm_0409_;
        '''
        with open('/SSD/lz/time_tsm/scripts/transfer_pretrain_finetune.sh', 'a') as f:
            f.write(
                'python train.py --backbone fcn --task classification --classifier linear --classifier_input 128 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                '--dataroot /SSD/lz/UCRArchive_2018 '
                '--dataset ' + target_dataset
                + ' --mode finetune --epoch 1000 --classifier linear' +
                ' --loss cross_entropy --source_dataset ' + source_dataset + ' --transfer_strategy classification '
                                                                             '--cuda cuda:1 --save_csv_name ' + source_dataset + '_finetune_cls_0409_' + ';\n')

i = 0
for source_dataset in source_datasets:  ## rec fcn finetune
    print("i = ", i, "dataset_name = ", source_dataset)
    i = i + 1
    for target_dataset in target_datasets:
        ### finetune rec fcn
        '''
        python train.py --backbone fcn --task classification --classifier linear --classifier_input 128 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --mode finetune --epoch 20  --loss cross_entropy --decoder_backbone fcn --source_dataset Coffee --transfer_strategy reconstruction --cuda cuda:1 --save_csv_name test_fcn_nonlin_single_norm_0409_;
        '''
        with open('/SSD/lz/time_tsm/scripts/transfer_pretrain_finetune.sh', 'a') as f:
            f.write(
                'python train.py --backbone fcn --task classification --classifier linear --classifier_input 128 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                '--dataroot /SSD/lz/UCRArchive_2018 '
                '--dataset ' + target_dataset
                + ' --mode finetune --epoch 1000 --classifier linear' +
                ' --loss cross_entropy --decoder_backbone fcn --source_dataset ' + source_dataset + ' --transfer_strategy reconstruction '
                                                                                                    '--cuda cuda:1 --save_csv_name ' + source_dataset + '_finetune_rec_fcn_0409_' + ';\n')

i = 0
for source_dataset in source_datasets:  ## rec rnn finetune
    print("i = ", i, "dataset_name = ", source_dataset)
    i = i + 1
    for target_dataset in target_datasets:
        ### finetune rec rnn
        '''
        python train.py --backbone fcn --task classification --classifier linear --classifier_input 128 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --mode finetune --epoch 20  --loss cross_entropy --decoder_backbone rnn --source_dataset Coffee --transfer_strategy reconstruction --cuda cuda:1 --save_csv_name test_fcn_nonlin_single_norm_0409_;
        '''
        with open('/SSD/lz/time_tsm/scripts/transfer_pretrain_finetune.sh', 'a') as f:
            f.write(
                'python train.py --backbone fcn --task classification --classifier linear --classifier_input 128 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                '--dataroot /SSD/lz/UCRArchive_2018 '
                '--dataset ' + target_dataset
                + ' --mode finetune --epoch 1000 --classifier linear' +
                ' --loss cross_entropy --decoder_backbone rnn --source_dataset ' + source_dataset + ' --transfer_strategy reconstruction '
                                                                                                    '--cuda cuda:1 --save_csv_name ' + source_dataset + '_finetune_rec_rnn_0409_' + ';\n')

## nohup ./scripts/transfer_pretrain_finetune.sh &
