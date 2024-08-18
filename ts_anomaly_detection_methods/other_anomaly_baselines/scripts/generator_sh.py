

uni_datasets = ['kpi', 'yahoo']
multi_datasets = ['SMD', 'MSL', 'SMAP', 'PSM', 'SWAT', 'NIPS_TS_Swan', 'NIPS_TS_Water']  ##  SMD, MSL, SMAP, PSM, SWAT, NIPS_TS_Swan, UCR, NIPS_TS_Water , 'UCR'



# code_main = 'main_gpt4ts_uea'   ## main_patchtst_ucr  main_gpt4ts_ucr  mian_patchtst

code_main_list = ['train_spot', 'train_dspot', 'train_lstm_vae', 'train_donut', 'train_ts2vec']


# for dataset in uni_datasets:
#     i = 1
#     for code_main in code_main_list:
#         print("i = ", i, "dataset_name = ", dataset)
#         i = i + 1
#
#         save_csv_name = code_main + '_0717.csv'  ##  --len_k
#
#         with open('/dev_data/lz/tsm_ptms_anomaly_detection/other_anomaly_baselines/scripts/uni_at.sh', 'a') as f:
#             f.write('python ' + code_main + '.py '
#                     '--dataset ' + dataset
#                     +
#                     ' --save_csv_name ' + save_csv_name + ' --gpu 0' + ';\n')


# for _index in range(1,251):
#     i = 1
#     for code_main in code_main_list:
#         print("i = ", i, "dataset_name = UCR")
#         i = i + 1
#
#         save_csv_name = code_main + '_ucr_0715.csv'  ##  --len_k
#
#         with open('/dev_data/lz/tsm_ptms_anomaly_detection/other_anomaly_baselines/scripts/ucr_at.sh', 'a') as f:
#             f.write('python ' + code_main + '_multi.py '
#                     '--dataset UCR --index ' + str(_index)
#                     +
#                     ' --save_csv_name ' + save_csv_name + ' --gpu 0' + ';\n')

# code_main_list = ['train_lstm_vae_multi', 'train_donut_multi', 'train_ts2vec_multi', 'train_dcdetector']
# for dataset in multi_datasets:
#     i = 1
#     for code_main in code_main_list:
#         print("i = ", i, "dataset_name = ", dataset)
#         i = i + 1
#
#         save_csv_name = code_main + '_0717.csv'  ##  --len_k
#
#         with open('/dev_data/lz/tsm_ptms_anomaly_detection/other_anomaly_baselines/scripts/multi_at.sh', 'a') as f:
#             f.write('python ' + code_main + '.py '
#                     '--dataset ' + dataset
#                     +
#                     ' --save_csv_name ' + save_csv_name + ' --gpu 0' + ';\n')


# code_main_list = ['train_timesnet', 'train_gpt4ts']
# for dataset in multi_datasets:
#     i = 1
#     for code_main in code_main_list:
#         print("i = ", i, "dataset_name = ", dataset)
#         i = i + 1
#
#         save_csv_name = code_main + '_0717.csv'  ##  --len_k
#
#         with open('/dev_data/lz/tsm_ptms_anomaly_detection/other_anomaly_baselines/scripts/multi_at.sh', 'a') as f:
#             f.write('python ' + code_main + '.py '
#                     '--data ' + dataset
#                     +
#                     ' --save_csv_name ' + save_csv_name + ' --gpu 0' + ';\n')
#

# code_main_list = ['train_at_multi']  ## , 'train_gpt4ts'  train_timesnet  train_dcdetector  train_at_multi
#
# for _index in range(1,251):
#     i = 1
#     for code_main in code_main_list:
#         print("i = ", i, "dataset_name = UCR")
#         i = i + 1
#
#         save_csv_name = code_main + '_ucr_0719.csv'  ##  --len_k
#
#         with open('/dev_data/lz/tsm_ptms_anomaly_detection/other_anomaly_baselines/scripts/ucr_at_zeta0.sh', 'a') as f:
#             f.write('python ' + code_main + '.py '
#                     '--anormly_ratio 0.5 --dataset UCR --index ' + str(_index)
#                     +
#                     ' --save_csv_name ' + save_csv_name + ' --cuda cuda:0' + ';\n')   ## anomaly_ratio  anormly_ratio  anormly_ratio


# code_main_list = ['train_dcdetector_nui']  ## , 'train_gpt4ts'  train_timesnet  train_dcdetector  train_at_multi
# ## train_gpt4ts_uni  train_timesnet_uni
# for dataset in uni_datasets:
#     i = 1
#     for code_main in code_main_list:
#         print("i = ", i, "dataset_name = UCR")
#         i = i + 1
#
#         save_csv_name = code_main + '_hm_0720.csv'  ##  --len_k
#
#         with open('/SSD/lz/tsm_ptms_anomaly_detection/other_anomaly_baselines/scripts/ucr_at.sh', 'a') as f:
#             f.write('python ' + code_main + '.py '
#                     '--anormly_ratio 1 --dataset ' + dataset
#                     +
#                     ' --save_csv_name ' + save_csv_name + ' --gpu 0' + ';\n')   ## anomaly_ratio  anormly_ratio  anormly_ratio


code_main_list = ['train_gpt4ts']  ## , 'train_gpt4ts'  train_timesnet  train_dcdetector  train_at_multi
## train_gpt4ts_uni  train_timesnet_uni
uni_datasets =  [79, 108, 187, 203]
for dataset in uni_datasets:
    i = 1
    for code_main in code_main_list:
        print("i = ", i, "dataset_name = UCR")
        i = i + 1

        # save_csv_name = code_main + '_hm_0720.csv'  ##  --len_k

        with open('/dev_data/lz/tsm_ptms_anomaly_detection/other_anomaly_baselines/scripts/ucr_at.sh', 'a') as f:
            f.write('python ' + code_main + '.py '
                    '--index ' + str(dataset)
                    +  ';\n')   ## anomaly_ratio  anormly_ratio  anormly_ratio


###  --cuda cuda:0

## nohup ./scripts/uni_at.sh &

## nohup ./scripts/multi_at.sh &

## nohup ./scripts/ucr_at.sh &

## nohup ./scripts/ucr_at_delta_0.sh &

## nohup ./scripts/ucr_at_delta_1.sh &

## nohup ./scripts/ucr_at_delta_1_2.sh &


## nohup ./scripts/ucr_at_zeta0.sh &

## nohup ./scripts/at_zeta1.sh &
## nohup ./scripts/at_zeta0.sh &

## nohup ./scripts/kpi.sh &
## nohup ./scripts/yahoo.sh &