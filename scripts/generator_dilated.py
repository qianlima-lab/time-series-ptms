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
    '''
        python train.py --backbone dilated --classifier linear --classifier_input 320 --depth 3 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset ACSF1 --mode directly_cls --epoch 10  --loss cross_entropy --save_csv_name test_nonlin_set_norm_0409_ --cuda cuda:1
        '''
    with open('/SSD/lz/time_tsm/scripts/dilated_single_norm.sh', 'a') as f:
        f.write('python train.py --backbone dilated --classifier linear --classifier_input 320 '
                '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                '--dataset ' + dataset
                + ' --mode directly_cls --epoch 1000 --depth 3 ' +
                ' --loss cross_entropy --save_csv_name dilated3_lin_single_norm_0409_ --cuda cuda:1' + ';\n')

i = 0
for dataset in ucr_dataset:
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1
    '''
    python train.py --backbone dilated --classifier nonlinear --classifier_input 320 --depth 3 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --mode directly_cls --epoch 10  --loss cross_entropy --save_csv_name test_nonlin_set_norm_0409_ --cuda cuda:1;
    '''
    with open('/SSD/lz/time_tsm/scripts/dilated_single_norm.sh', 'a') as f:
        f.write('python train.py --backbone dilated --classifier nonlinear --classifier_input 320 '
                '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                '--dataset ' + dataset
                + ' --mode directly_cls --epoch 1000 --depth 3 ' +
                ' --loss cross_entropy --save_csv_name dilated3_nonlin_single_norm_0409_ --cuda cuda:1' + ';\n')

i = 0
for dataset in ucr_dataset:
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1
    '''
        python train.py --backbone dilated --classifier linear --classifier_input 320 --depth 10 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset ACSF1 --mode directly_cls --epoch 10  --loss cross_entropy --save_csv_name test_nonlin_set_norm_0409_ --cuda cuda:1
        '''
    with open('/SSD/lz/time_tsm/scripts/dilated_single_norm.sh', 'a') as f:
        f.write('python train.py --backbone dilated --classifier linear --classifier_input 320 '
                '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                '--dataset ' + dataset
                + ' --mode directly_cls --epoch 1000 --depth 10 ' +
                ' --loss cross_entropy --save_csv_name dilated10_lin_single_norm_0409_ --cuda cuda:1' + ';\n')

i = 0
for dataset in ucr_dataset:
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1
    '''
    python train.py --backbone dilated --classifier nonlinear --classifier_input 320 --depth 10 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way single --dataset Coffee --mode directly_cls --epoch 10  --loss cross_entropy --save_csv_name test_nonlin_set_norm_0409_ --cuda cuda:1;
    '''
    with open('/SSD/lz/time_tsm/scripts/dilated_single_norm.sh', 'a') as f:
        f.write('python train.py --backbone dilated --classifier nonlinear --classifier_input 320 '
                '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                '--dataset ' + dataset
                + ' --mode directly_cls --epoch 1000 --depth 10 ' +
                ' --loss cross_entropy --save_csv_name dilated10_nonlin_single_norm_0409_ --cuda cuda:1' + ';\n')


## nohup ./scripts/dilated_single_norm.sh &