from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
from other_anomaly_baselines.models import TimesNet
from other_anomaly_baselines.models import GPT4TS

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import math
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




def adjustment(gt, pred):
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
    return gt, pred


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'GPT4TS': GPT4TS,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args, train_set, train_loader, val_set, val_loader, test_set, test_loader):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.train_set = train_set
        self.train_loader = train_loader
        self.val_set = val_set
        self.val_loader = val_loader
        self.test_set = test_set
        self.test_loader = test_loader

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # data_set, data_loader = data_provider(self.args, flag)
        if flag == 'train':
            return self.train_set, self.train_loader

        if flag == 'val':
            return self.val_set, self.val_loader

        if flag == 'test':
            return self.test_set, self.test_loader

        # return self.data_set, self.data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def vali_uni(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_x in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                # print("loss = ", loss)
                # print("batch_x.shape = ", batch_x.shape, ", outputs.shape = ", outputs.shape)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def train_uni(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                # print("batch_x.shape = ", batch_x.shape, ", batch_x[:5] = ", batch_x[:5])

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                # print("loss = ", loss)
                # print("batch_x.shape = ", batch_x.shape, ", outputs.shape = ", outputs.shape)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali_uni(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, dataset=None, ucr_index=None):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # if dataset == 'UCR':
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

        # else:
        #
        #     results_f1_pa_k_10 = evaluate.evaluate(test_energy, test_labels, k=10)
        #     results_f1_pa_k_50 = evaluate.evaluate(test_energy, test_labels, k=50)
        #     results_f1_pa_k_90 = evaluate.evaluate(test_energy, test_labels, k=90)
        #
        #     eval_res = {
        #         'f1': None,
        #         'precision': None,
        #         'recall': None,
        #         "Affiliation precision": None,
        #         "Affiliation recall": None,
        #         "R_AUC_ROC": None,
        #         "R_AUC_PR": None,
        #         "VUS_ROC": None,
        #         "VUS_PR": None,
        #         'f1_pa_10': results_f1_pa_k_10['best_f1_w_pa'],
        #         'f1_pa_50': results_f1_pa_k_50['best_f1_w_pa'],
        #         'f1_pa_90': results_f1_pa_k_90['best_f1_w_pa'],
        #     }
        if ucr_index == 79 or ucr_index == 108 or ucr_index == 187 or ucr_index == 203:
            pass
        else:

            if dataset == 'SMD' or dataset == 'NIPS_TS_Swan' or dataset == 'NIPS_TS_Water' or dataset == 'SWAT':
                pass
            else:
                scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
                for key, value in scores_simple.items():
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


        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        eval_res['f1'] = f_score
        eval_res['precision'] = precision
        eval_res['recall'] = recall


        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()


        return eval_res


    def test_uni(self, setting, all_train_data, all_test_data, all_test_labels, all_test_timestamps, delay, config, test=0):
        # test_data, test_loader = self._get_data(flag='test')
        # train_data, train_loader = self._get_data(flag='train')


        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set

        with torch.no_grad():
            # for i, (batch_x, batch_y) in enumerate(train_loader):
            #     batch_x = batch_x.float().to(self.device)
            #     # reconstruction
            #     outputs = self.model(batch_x, None, None, None)
            #     # criterion
            #     score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            #     score = score.detach().cpu().numpy()
            #     attens_energy.append(score)

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
                    # print("type(input) = ", type(input_data), input_data.shape)
                    batch_x = input_data.float().to(self.device)
                    # reconstruction
                    outputs = self.model(batch_x, None, None, None)
                    # criterion
                    score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                    score = score.detach().cpu().numpy()
                    attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        with torch.no_grad():

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
                    batch_x = input_data.float().to(self.device)

                    outputs = self.model(batch_x, None, None, None)
                    # criterion
                    score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                    score = score.detach().cpu().numpy()
                    attens_energy.append(score)
                    test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # if dataset == 'UCR':
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

        # scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        # for key, value in scores_simple.items():
        #     if key == 'Affiliation precision':
        #         eval_res["Affiliation precision"] = value
        #     if key == 'Affiliation recall':
        #         eval_res["Affiliation recall"] = value
        #     if key == 'R_AUC_ROC':
        #         eval_res["R_AUC_ROC"] = value
        #     if key == 'R_AUC_PR':
        #         eval_res["R_AUC_PR"] = value
        #     if key == 'VUS_ROC':
        #         eval_res["VUS_ROC"] = value
        #     if key == 'VUS_PR':
        #         eval_res["VUS_PR"] = value


        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        eval_res['f1'] = f_score
        eval_res['precision'] = precision
        eval_res['recall'] = recall


        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()


        return eval_res
