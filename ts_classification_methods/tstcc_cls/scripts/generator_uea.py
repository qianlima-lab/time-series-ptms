uea_all = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
           'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing',
           'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting',
           'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
           'NATOPS', 'PenDigits', 'PEMS-SF', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
           'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

i = 1
for dataset in uea_all:
    print("i = ", i, ", dataset = ", dataset)
    ## python main_uea.py --dataset BasicMotions  --device cuda:0 --save_csv_name tstcc_uea_0423_ --seed 42
    with open('/SSD/lz/time_tsm/tstcc_cls/scripts/fivefold_tstcc_uea.sh', 'a') as f:
        f.write(
            'python main_uea.py --dataset ' + dataset + ' --device cuda:0 --save_csv_name tstcc_uea_0423_ --seed 42' + ';\n')

    i = i + 1

    ## nohup ./scripts/fivefold_tstcc_uea.sh &
