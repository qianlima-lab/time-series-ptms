## Crop PLAID PigAirwayPressure Phoneme PickupGestureWiimoteZ

ucr_dataset = ['Crop', 'PLAID', 'PigAirwayPressure', 'Phoneme', 'PickupGestureWiimoteZ']

i = 0
for dataset in ucr_dataset:
    print("i = ", i, "dataset_name = ", dataset)
    i = i + 1
    '''
    python train.py --backbone fcn --classifier nonlinear --classifier_input 128 --dataroot /SSD/lz/UCRArchive_2018 --normalize_way train_set --dataset ACSF1 --mode directly_cls --epoch 1000  --loss cross_entropy --save_csv_name fcn_nonlin_set_norm_0404_ --cuda cuda:1
    '''
    with open('/SSD/lz/time_tsm/scripts/fcn_nonlin_single_seed.sh', 'a') as f:
        seeds = [2, 15, 21, 37, 47, 53, 66, 73, 87, 99]
        for seed in seeds:
            f.write('python train.py --backbone fcn --classifier nonlinear --classifier_input 128 '
                    '--dataroot /SSD/lz/UCRArchive_2018 --normalize_way single '
                    '--dataset ' + dataset + ' --random_seed ' + str(seed)
                    + ' --mode directly_cls --epoch 1000 ' +
                    ' --loss cross_entropy --save_csv_name fcn_nonlin_single_seed_0407_ --cuda cuda:1' + ';\n')

## nohup ./scripts/fcn_nonlin_single_seed.sh &