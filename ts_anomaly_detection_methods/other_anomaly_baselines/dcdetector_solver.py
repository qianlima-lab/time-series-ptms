import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
# from utils.utils import *
from other_anomaly_baselines.models.DCdetector import DCdetector
from other_anomaly_baselines.datasets.data_loader import get_loader_segment
from einops import rearrange
from other_anomaly_baselines.metrics.metrics import *
import warnings
from tadpak import evaluate
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')


class UniLoader_train(object):
    def __init__(self, data_set, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size

        self.train = data_set


    def __len__(self):
        """
        Number of images in the object dataset.
        """

        return (self.train.shape[0] - self.win_size) // self.step + 1


    def __getitem__(self, index):
        index = index * self.step

        return np.float32(self.train[index:index + self.win_size])


class UniLoader_test(object):
    def __init__(self, data_set, label_set, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size

        self.train = data_set
        self.train_labels = label_set


    def __len__(self):
        """
        Number of images in the object dataset.
        """

        return (self.train.shape[0] - self.win_size) // self.step + 1


    def __getitem__(self, index):
        index = index * self.step

        return np.float32(self.train[index:index + self.win_size]), np.float32(self.train_labels[0:self.win_size])




def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0, index=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.index = index

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        print("os.path.join(path, str(self.dataset) + '_checkpoint.pth') = ", os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + str(self.index) +'_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config, multi=True):

        self.__dict__.update(Solver.DEFAULTS, **config)

        if multi:
            self.train_loader, _ = get_loader_segment(self.index, self.data_path + self.dataset, batch_size=self.batch_size,
                                                   win_size=self.win_size, mode='train', dataset=self.dataset, )
            self.vali_loader, _ = get_loader_segment(self.index, self.data_path + self.dataset, batch_size=self.batch_size,
                                                  win_size=self.win_size, mode='val', dataset=self.dataset)
            self.test_loader, _ = get_loader_segment(self.index, self.data_path + self.dataset, batch_size=self.batch_size,
                                                  win_size=self.win_size, mode='test', dataset=self.dataset)
            self.thre_loader, _ = get_loader_segment(self.index, self.data_path + self.dataset, batch_size=self.batch_size,
                                                  win_size=self.win_size, mode='thre', dataset=self.dataset)
        else:
            self.train_loader, _ = None, None
            self.vali_loader, _ = None, None
            self.test_loader, _ = None, None
            self.thre_loader, _ = None, None

        self.build_model()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads,
                                d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size,
                                channel=self.input_c)

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.dataset, index=self.index)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            # for i, data in enumerate(self.train_loader):
            #     print(data)
            #     break

            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)


                # print("input = ", type(input), input.shape)
                series, prior = self.model(input)

                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self, ucr_index=None):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + str(self.index) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        # labels = np.asarray(labels_log, np.int64)[0]

        # print("test_energy.shape = ", test_energy.shape, test_labels.shape)
        # print("test_energy.shape = ", test_energy[:10])
        # print("test_labels.shape = ", test_labels[:10])
        index_list =  [38, 54, 71, 72, 79, 85, 88, 108, 146, 162, 179, 180, 187, 193, 196, 203, 212, 229, 232]
        if ucr_index in index_list:
            eval_res = {
                'f1': None,
                'precision': None,
                'recall': None,
                "Affiliation precision": None,
                "Affiliation recall": None,
                "R_AUC_ROC": None,
                "R_AUC_PR": None,
                "VUS_ROC": None,
                "VUS_PR": None,
                'f1_pa_10': None,
                'f1_pa_50': None,
                'f1_pa_90': None,
            }
        else:

            results_f1_pa_k_10 = evaluate.evaluate(test_energy, test_labels, k=10)
            results_f1_pa_k_50 = evaluate.evaluate(test_energy, test_labels, k=50)
            results_f1_pa_k_90 = evaluate.evaluate(test_energy, test_labels, k=90)

            eval_res = {
                'f1': None,
                'precision': None,
                'recall': None,
                "Affiliation precision": None,
                "Affiliation recall": None,
                "R_AUC_ROC": None,
                "R_AUC_PR": None,
                "VUS_ROC": None,
                "VUS_PR": None,
                'f1_pa_10': results_f1_pa_k_10['best_f1_w_pa'],
                'f1_pa_50': results_f1_pa_k_50['best_f1_w_pa'],
                'f1_pa_90': results_f1_pa_k_90['best_f1_w_pa'],
            }

            matrix = [self.index]
            scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
            for key, value in scores_simple.items():
                matrix.append(value)
                if key == 'Affiliation precision':
                    eval_res["Affiliation precision"] = value
                if key == 'Affiliation recall':
                    eval_res["Affiliation recall"] = value
                if key == 'R_AUC_ROC':
                    eval_res["R_AUC_ROC"] = value
                if key == 'R_AUC_PR':
                    eval_res["R_AUC_PR"] = value
                if key == 'VUS_ROC':
                    eval_res["VUS_ROC"] = value
                if key == 'VUS_PR':
                    eval_res["VUS_PR"] = value

                print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        # if self.data_path == 'UCR' or 'UCR_AUG':
        #     import csv
        #     with open('result_dc/' + self.dataset + '.csv', 'a+') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(matrix)

        eval_res['f1'] = f_score
        eval_res['precision'] = precision
        eval_res['recall'] = recall

        return eval_res

    def vali_uni(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, input_data in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train_uni(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.dataset, index=self.index)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            # for i, data in enumerate(self.train_loader):
            #     print(data)
            #     break

            for i, input_data in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)


                # print("input = ", type(input), input.shape)
                series, prior = self.model(input)

                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali_uni(self.vali_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test_uni(self, all_train_data, all_test_data, all_test_labels, all_test_timestamps, delay, config):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + str(self.index) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []

        for k in all_train_data:
            train_data = all_train_data[k]

            train_data = np.array(train_data)

            # train_data =
            train_data = np.expand_dims(train_data, axis=-1)
            train_dataset = UniLoader_train(train_data, config.win_size, 1)

            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=2,
                                      drop_last=True)

            for i, input_data in enumerate(train_loader):
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for k in all_train_data:
            _test_labels = all_test_labels[k]
            test_data = all_test_data[k]

            test_data = np.array(test_data)

            test_data = np.expand_dims(test_data, axis=-1)

            test_dataset = UniLoader_test(test_data, _test_labels, config.win_size, 1)

            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     num_workers=2,
                                     drop_last=True)

            for i, (input_data, labels) in enumerate(test_loader):
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for k in all_train_data:
            _test_labels = all_test_labels[k]
            test_data = all_test_data[k]

            test_data = np.array(test_data)

            test_data = np.expand_dims(test_data, axis=-1)

            test_dataset = UniLoader_test(test_data, _test_labels, config.win_size, 1)

            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     num_workers=2,
                                     drop_last=True)
            for i, (input_data, labels) in enumerate(test_loader):
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        # labels = np.asarray(labels_log, np.int64)[0]

        # print("test_energy.shape = ", test_energy.shape, test_labels.shape)
        # print("test_energy.shape = ", test_energy[:10])
        # print("test_labels.shape = ", test_labels[:10])

        # results_f1_pa_k_10 = evaluate.evaluate(test_energy, test_labels, k=10)
        # results_f1_pa_k_50 = evaluate.evaluate(test_energy, test_labels, k=50)
        # results_f1_pa_k_90 = evaluate.evaluate(test_energy, test_labels, k=90)

        eval_res = {
            'f1': None,
            'precision': None,
            'recall': None,
            "Affiliation precision": None,
            "Affiliation recall": None,
            "R_AUC_ROC": None,
            "R_AUC_PR": None,
            "VUS_ROC": None,
            "VUS_PR": None,
            # 'f1_pa_10': results_f1_pa_k_10['best_f1_w_pa'],
            # 'f1_pa_50': results_f1_pa_k_50['best_f1_w_pa'],
            # 'f1_pa_90': results_f1_pa_k_90['best_f1_w_pa'],
        }

        # matrix = [self.index]

        min_len = min(min(pred.shape[0], gt.shape[0]), test_energy.shape[0])

        scores_simple = combine_all_evaluation_scores(pred[:min_len], gt[:min_len], test_energy[:min_len])
        for key, value in scores_simple.items():
            # matrix.append(value)
            if key == 'Affiliation precision':
                eval_res["Affiliation precision"] = value
            if key == 'Affiliation recall':
                eval_res["Affiliation recall"] = value
            if key == 'R_AUC_ROC':
                eval_res["R_AUC_ROC"] = value
            if key == 'R_AUC_PR':
                eval_res["R_AUC_PR"] = value
            if key == 'VUS_ROC':
                eval_res["VUS_ROC"] = value
            if key == 'VUS_PR':
                eval_res["VUS_PR"] = value

            print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt[:min_len], pred[:min_len], average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        # if self.data_path == 'UCR' or 'UCR_AUG':
        #     import csv
        #     with open('result_dc/' + self.dataset + '.csv', 'a+') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(matrix)

        eval_res['f1'] = f_score
        eval_res['precision'] = precision
        eval_res['recall'] = recall

        return eval_res
