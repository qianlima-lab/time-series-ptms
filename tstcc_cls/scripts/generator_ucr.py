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

i = 1
for dataset in ucr_dataset:
    print("i = ", i, ", dataset = ", dataset)
    ## python main_ucr.py --dataset Coffee  --device cuda:1 --save_csv_name tstcc_0327_ --seed 42
    with open('/SSD/lz/time_tsm/tstcc_cls/scripts/fivefold_tstcc_ucr.sh', 'a') as f:
        f.write(
            'python main_ucr.py --dataset ' + dataset + ' --device cuda:0 --save_csv_name tstcc_ucr_0423_ --seed 42' + ';\n')

    i = i + 1

    ## nohup ./scripts/fivefold_tstcc_ucr.sh &
