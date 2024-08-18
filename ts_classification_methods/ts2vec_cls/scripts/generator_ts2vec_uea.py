uea_all = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
           'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing',
           'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting',
           'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
           'NATOPS', 'PenDigits', 'PEMS-SF', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
           'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

i = 0
for dataset in uea_all:
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1
    '''
          python train_tsm_uea.py --dataroot /SSD/lz/Multivariate2018_arff --dataset BasicMotions --gpu 1 --batch-size 8 --save_csv_name ts2vec_tsm_uea_0423_
       '''
    with open('/SSD/lz/time_tsm/ts2vec_cls/scripts/ts2vec_tsm_uea.sh', 'a') as f:
        f.write('python train_tsm_uea.py '
                '--dataroot /SSD/lz/Multivariate2018_arff '
                '--dataset ' + dataset
                + ' --gpu 1 --batch-size 8 ' +
                ' --save_csv_name ts2vec_tsm_uea_0423_' + ';\n')

## nohup ./scripts/ts2vec_tsm_uea.sh &
