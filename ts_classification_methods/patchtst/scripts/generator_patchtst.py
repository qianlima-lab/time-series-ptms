uea_all = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
           'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing',
           'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting',
           'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
           'NATOPS', 'PenDigits', 'PEMS-SF', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
           'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

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

code_main = 'main_gpt4ts_uea'   ## main_patchtst_ucr  main_gpt4ts_ucr  mian_patchtst

_mu_all = [8]  ## , 16, 24, 32, 48, 64

i = 1
for dataset in uea_all:
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1

    save_csv_name = code_main + '_0702_'  ##  --len_k

    with open('/dev_data/lz/time_series_label_noise/patchtst/scripts/patchtst_ucr.sh', 'a') as f:
        f.write('python ' + code_main + '.py '
                '--dataset ' + dataset
                + ' --epoch 1000 '+
                '--save_csv_name ' + save_csv_name + ' --cuda cuda:1' + ';\n')

## nohup ./scripts/patchtst_uea.sh &

## nohup ./scripts/patchtst_ucr.sh &