import argparse
import os
import sys
import time
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import torch

from data.preprocessing import k_fold, load_UEA, fill_nan_value, normalize_uea_set
from tsm_utils import save_cls_result, set_seed
from tstcc_cls.models.TC import TC
from tstcc_cls.models.model import base_Model
from tstcc_cls.trainer.trainer import Trainer_cls
from tstcc_cls.utils import _logger, generator_uea_config, generator_uea

# Args selections
start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=42, type=int,
                    help='seed value')
parser.add_argument('--random_seed', type=int, default=42, help='The random seed')
parser.add_argument('--training_mode', default='self_supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='uea', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')  ## HAR
parser.add_argument('--dataset', default='EigenWorms', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda:1', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--save_csv_name', type=str, default='test_tstcc_uea_0425_')
parser.add_argument('--save_dir', type=str, default='/SSD/lz/time_tsm/tstcc_cls/result')
args = parser.parse_args()
set_seed(args)

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = f"./data/{data_type}"

sum_dataset, sum_target, num_classes = load_UEA(
    dataroot='/SSD/lz/Multivariate2018_arff',
    dataset=args.dataset)
# sum_dataset = sum_dataset[..., np.newaxis]
train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = k_fold(
    sum_dataset, sum_target)
# print("Start features_len = ", configs.features_len, ", num_classes = ", configs.num_classes)
generator_uea_config(data=train_datasets[0], label=train_targets[0], configs=configs)
if args.dataset == 'EigenWorms':
    configs.augmentation.max_seg = 5
    configs.batch_size = 8
if train_datasets[0].shape[1] <= 30:
    configs.TC.timesteps = 1
# print("End features_len = ", configs.features_len, ", num_classes = ", configs.num_classes, ", input_channels = ",
#       configs.input_channels)
# train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
train_accuracies = []
val_accuracies = []
test_accuracies = []
t = time.time()
for i in range(5):
    ### mean impute
    train_data, val_data, test_data = fill_nan_value(train_datasets[i], val_datasets[i], test_datasets[i])

    ### normalize
    train_data = normalize_uea_set(train_data)
    val_data = normalize_uea_set(val_data)
    test_data = normalize_uea_set(test_data)

    # train_data = train_data[..., np.newaxis]
    # val_data = val_data[..., np.newaxis]
    # test_data = test_data[..., np.newaxis]

    train_dl = generator_uea(data=train_data, label=train_targets[i],
                             configs=configs, training_mode='self_supervised', drop_last=True)
    valid_dl = generator_uea(data=val_data, label=val_targets[i],
                             configs=configs, training_mode='self_supervised', drop_last=False)
    test_dl = generator_uea(data=test_data, label=test_targets[i],
                            configs=configs, training_mode='self_supervised', drop_last=False)
    logger.debug("Data loaded ...")

    # Load Model
    model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                                betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

    # copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type) # to do it only once

    # self_supervised Trainer
    Trainer_cls(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl,
                test_dl, device, logger, configs, experiment_log_dir, training_mode='self_supervised')
    print("Self_supervised end, start fine_tune!")
    # fine_tune Trainer
    train_dl = generator_uea(data=train_data, label=train_targets[i],
                             configs=configs, training_mode='fine_tune', drop_last=True)
    valid_dl = generator_uea(data=val_data, label=val_targets[i],
                             configs=configs, training_mode='fine_tune', drop_last=False)
    test_dl = generator_uea(data=test_data, label=test_targets[i],
                            configs=configs, training_mode='fine_tune', drop_last=False)

    train_acc, val_acc, test_acc = Trainer_cls(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer,
                                               train_dl, valid_dl,
                                               test_dl, device, logger, configs, experiment_log_dir,
                                               training_mode='fine_tune')
    # print(type(train_acc.data), train_acc.numpy(), val_acc.numpy(), test_acc.numpy())
    # train_accuracies = torch.Tensor(train_accuracies)
    # test_accuracies = torch.Tensor(test_accuracies)
    train_accuracies.append(train_acc.item())
    val_accuracies.append(val_acc.item())
    test_accuracies.append(test_acc.item())

train_time = time.time() - t
print("train_accuracies = ", train_accuracies, len(train_accuracies))
test_accuracies = torch.Tensor(test_accuracies)
save_cls_result(args, test_accu=torch.mean(test_accuracies), test_std=torch.std(test_accuracies),
                train_time=train_time / 5, end_val_epoch=0.0, seeds=args.seed)
logger.debug(f"Training time is : {datetime.now() - start_time}")
