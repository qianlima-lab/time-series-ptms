ucr_dataset = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'ArrowHead', 'BME',
               'Beef',
               'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee',
               'Computers',
               'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup',
               'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend',
               'ECG200', 'ECG5000', 'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes',
               'ElectricDevices',
               'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB',
               'FreezerRegularTrain',
               'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1',
               'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
               'Ham',
               'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain',
               'InsectEPGSmallTrain',
               'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
               'Mallat', 'Meat',
               'MedicalImages', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
               'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain',
               'NonInvasiveFetalECGThorax1',
               'NonInvasiveFetalECGThorax2', 'OSULeaf', 'OliveOil', 'PLAID', 'PhalangesOutlinesCorrect', 'Phoneme',
               'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'Plane', 'PowerCons',
               'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
               'RefrigerationDevices',
               'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2',
               'ShakeGestureWiimoteZ',
               'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
               'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
               'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD',
               'UWaveGestureLibraryAll',
               'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms',
               'Worms',
               'WormsTwoClass', 'Yoga']

i = 0
for dataset in ucr_dataset:
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1
    # '''
    # python train_fcn.py --backbone fcn --classifier nonlinear --classifier_input 128 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way train_set --dataset Coffee --fcn_epoch 1000 --gpu 1 --batch-size 8 --loss cross_entropy --save_csv_name ts2vec_fcn_set_norm_0404_ --cuda cuda:1
    # '''
    # with open('/SSD/lz/time_tsm/ts2vec_cls/scripts/ts2vec_fcn_set_norm.sh', 'a') as f:
    #     f.write('python train_fcn.py --backbone fcn --classifier nonlinear --classifier_input 128 '
    #             '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way train_set '
    #             '--dataset ' + dataset
    #             + ' --fcn_epoch 1000 --gpu 1 --batch-size 8 ' +
    #             ' --loss cross_entropy --save_csv_name ts2vec_fcn_set_norm_0404_ --cuda cuda:1' + ';\n')
    #
    # '''
    # python train_fcn.py --backbone fcn --classifier nonlinear --classifier_input 128 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single  --dataset Coffee --fcn_epoch 1000 --gpu 1 --batch-size 8 --loss cross_entropy --save_csv_name ts2vec_fcn_single_norm_0404_ --cuda cuda:1
    # '''
    # with open('/SSD/lz/time_tsm/ts2vec_cls/scripts/ts2vec_fcn_single_norm.sh', 'a') as f:
    #     f.write('python train_fcn.py --backbone fcn --classifier nonlinear --classifier_input 128 '
    #             '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
    #             '--dataset ' + dataset
    #             + ' --fcn_epoch 1000 --gpu 1 --batch-size 8 ' +
    #             ' --save_csv_name ts2vec_fcn_single_norm_0404_ --cuda cuda:1' + ';\n')


    # '''
    # python train_tsm.py --dataroot /SSD/lz/UCRArchive_2018 --normalize_way train_set --dataset Coffee --gpu 1 --batch-size 8 --save_csv_name ts2vec_tsm_set_norm_0404_
    # '''
    # with open('/SSD/lz/time_tsm/ts2vec_cls/scripts/ts2vec_tsm_set_norm.sh', 'a') as f:
    #      f.write('python train_tsm.py '
    #             '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way train_set '
    #             '--dataset ' + dataset
    #             + ' --gpu 1 --batch-size 8 ' +
    #             ' --save_csv_name ts2vec_tsm_set_norm_0404_' + ';\n')
    #
    '''
       python train_tsm.py --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --gpu 1 --batch-size 8 --save_csv_name ts2vec_tsm_single_norm_0404_
    '''
    with open('/SSD/lz/time_tsm/ts2vec_cls/scripts/ts2vec_tsm_single_norm.sh', 'a') as f:
        f.write('python train_tsm.py '
                '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                '--dataset ' + dataset
                + ' --gpu 1 --batch-size 8 ' +
                ' --save_csv_name ts2vec_tsm_train_val_b8_single_norm_0409_' + ';\n')


i = 0
for dataset in ucr_dataset:
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1
    '''
          python train_tsm.py --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --gpu 1 --batch-size 16 --save_csv_name ts2vec_tsm_single_norm_0404_
       '''
    with open('/SSD/lz/time_tsm/ts2vec_cls/scripts/ts2vec_tsm_single_norm.sh', 'a') as f:
        f.write('python train_tsm.py '
                '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                '--dataset ' + dataset
                + ' --gpu 1 --batch-size 16 ' +
                ' --save_csv_name ts2vec_tsm_train_val_b16_single_norm_0409_' + ';\n')


## nohup ./scripts/ts2vec_fcn_set_norm.sh &
## nohup ./scripts/ts2vec_fcn_single_norm.sh &

## nohup ./scripts/ts2vec_tsm_set_norm.sh &
## nohup ./scripts/ts2vec_tsm_single_norm.sh &