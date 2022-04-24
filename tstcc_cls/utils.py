import logging
import os
import random
import sys
from shutil import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

from tstcc_cls.dataloader.dataloader import Load_Dataset


def generator_ucr_config(data, label, configs):
    X = np.reshape(data, (data.shape[0], -1))
    Y = label
    num_class = np.unique(Y).shape[0]
    series_len = X.shape[1]
    for i in range(3):
        if series_len % 2 == 1:
            series_len = series_len + 3
            series_len = series_len // 2
        else:
            series_len = series_len // 2 + 1

    configs.features_len = series_len
    configs.num_classes = num_class

    while X.shape[0] < configs.batch_size:
        configs.batch_size = configs.batch_size // 2
    # print("num_class = ", num_class, ", features_len = ", features_len)


def generator_ucr(data, label, configs, training_mode, drop_last=True):
    # print("Raw data shape = ", data.shape)
    data = np.reshape(data, (data.shape[0], -1))
    # print("New data shape = ", data.shape)
    data_dict = dict()
    data_dict["samples"] = torch.from_numpy(data).unsqueeze(1)
    # print("samples data shape = ", data_dict["samples"].shape)
    data_dict["labels"] = torch.from_numpy(label)

    tr_dataset = Load_Dataset(data_dict, configs, training_mode)

    tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=configs.batch_size,
                                            shuffle=True, drop_last=drop_last,
                                            num_workers=0)

    return tr_loader


def generator_uea_config(data, label, configs):
    Y = label
    num_class = np.unique(Y).shape[0]
    series_len = data.shape[1]
    for i in range(3):
        if series_len % 2 == 1:
            series_len = series_len + 3
            series_len = series_len // 2
        else:
            series_len = series_len // 2 + 1

    configs.features_len = series_len
    configs.num_classes = num_class
    configs.input_channels = data.shape[2]

    while data.shape[0] < configs.batch_size:
        configs.batch_size = configs.batch_size // 2


def generator_uea(data, label, configs, training_mode, drop_last=True):
    data_dict = dict()
    print("shape = ", data.shape)
    data_dict["samples"] = torch.from_numpy(data)
    data_dict["labels"] = torch.from_numpy(label)

    tr_dataset = Load_Dataset(data_dict, configs, training_mode)

    tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=configs.batch_size,
                                            shuffle=True, drop_last=drop_last,
                                            num_workers=0)

    return tr_loader


def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))
