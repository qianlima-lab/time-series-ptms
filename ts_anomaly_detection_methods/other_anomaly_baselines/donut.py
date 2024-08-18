import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.donut_model import DONUT_Model
from utils import split_with_nan, centerize_vary_length_series
import math
import time
from tasks.anomaly_detection import eval_ad_result, np_shift
import bottleneck as bn
from sklearn.metrics import f1_score, precision_score, recall_score
from other_anomaly_baselines.metrics.affiliation.metrics import pr_from_events
from other_anomaly_baselines.metrics.vus.metrics import get_range_vus_roc
from other_anomaly_baselines.metrics.affiliation.generics import convert_vector_to_events
from tadpak import evaluate


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


class DONUT:
    
    def __init__(
        self,
        input_dims,
        latent_dim=100,
        hidden_dim=3,
        device='cuda',
        lr=0.001,
        batch_size=8,
        z_kld_weight=0.1,
        x_kld_weight=0.1,
        max_train_length=None,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.z_kld_weight = z_kld_weight
        self.x_kld_weight = x_kld_weight
        self.max_train_length = max_train_length
        self.input_dims = input_dims
       
        self.net = DONUT_Model(in_channel=input_dims, latent_dim=latent_dim, hidden_dim=hidden_dim).to(self.device)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    
    def train(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' 
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
                # train_data: (n_instance*sections, max_train_length, n_features)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0) # (max_train_length)
        if temporal_missing[0] or temporal_missing[-1]: # whether the head or tail exists nan
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)] 
        # delete the sequence (max_train_length, n_features) contains only nan

        for i in range(train_data.shape[0]):
            train_data[i][np.isnan(train_data[i])] = np.nanmean(train_data[i])
        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x = batch[0]  #(batch_size, n_timestamps, n_features)
                # print("#####################")
                # raise Exception('my personal exception!')

                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                
                optimizer.zero_grad()
                
                outputs, z_mu, z_log_var, x_mu, x_log_var = self.net(x) 
                loss = self.net.loss_function(x, outputs, z_mu, z_log_var, x_mu, x_log_var, self.z_kld_weight, self.x_kld_weight)
                
                loss.backward()
                optimizer.step()
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log
    

    def anomaly_score(self, model, test_data, is_multi=False):
        if is_multi:
            test_data = torch.from_numpy(np.float32(test_data.reshape(1, -1, self.input_dims))).to(self.device)
        else:
            test_data = torch.from_numpy(np.float32(test_data.reshape(1, -1, 1))).to(self.device)
        # test_data = torch.from_numpy(np.float32(test_data.reshape(1, -1, 1))).to(self.device)

        if self.max_train_length is not None and test_data.size(1) > self.max_train_length:
            window_offset = np.random.randint(test_data.size(1) - self.max_train_length + 1)
            test_data = test_data[:, window_offset: window_offset + self.max_train_length]

        # 设置批次大小
        batch_size = 2

        # 创建 DataLoader
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        self.net.eval()
        with torch.no_grad():
            # 初始化保存输出的列表
            outputs_list = []
            # z_mu_list = []
            # z_log_var_list = []
            # x_mu_list = []
            # x_log_var_list = []
            for input_data in test_loader:
                input_data = input_data[0]  # 从 TensorDataset 中提取数据


                # x = x.to(self.device)

                print("input_data.shape = ", input_data.shape)
                batch_outputs, batch_z_mu, batch_z_log_var, batch_x_mu, batch_x_log_var =  self.net(input_data)

                # 保存每个批次的结果
                outputs_list.append(batch_outputs)
                # z_mu_list.append(batch_z_mu)
                # z_log_var_list.append(batch_z_log_var)
                # x_mu_list.append(batch_x_mu)
                # x_log_var_list.append(batch_x_log_var)

            # 将所有批次结果整合
            outputs = torch.cat(outputs_list, dim=0)
            # z_mu = torch.cat(z_mu_list, dim=0)
            # z_log_var = torch.cat(z_log_var_list, dim=0)
            # x_mu = torch.cat(x_mu_list, dim=0)
            # x_log_var = torch.cat(x_log_var_list, dim=0)
            # print("test_data.shape = ", test_data.shape)
            # print("self.net = ", self.net)
            # outputs, z_mu, z_log_var, x_mu, x_log_var = self.net(test_data)

            # rec_error = torch.sum(torch.abs(outputs - test_data), dim=-1)
            rec_error = torch.sum(torch.square(outputs - test_data), dim=-1)
            rec_error = torch.flatten(rec_error)

        return rec_error
    
    def evaluate(self, model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay, is_multi=False, ucr_index=None):
        t = time.time()

        res_log = []
        labels_log = []
        timestamps_log = []
        res_log_socres = []
        if is_multi:
            train_data = all_train_data

            test_data = all_test_data
            test_labels = all_test_labels

            print("train_data.shape = ", train_data.shape, ", test_data.shape = ", test_data.shape)

            train_err = self.anomaly_score(model, train_data, is_multi=is_multi).detach().cpu().numpy()
            test_err = self.anomaly_score(model, test_data, is_multi=is_multi).detach().cpu().numpy()

            ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
            train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
            test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
            train_err_adj = train_err_adj[22:]

            thr = np.mean(train_err_adj) + 4 * np.std(train_err_adj)
            test_res = (test_err_adj > thr) * 1
            res_log_socres.append(test_err_adj)

            for i in range(len(test_res)):
                if i >= delay and test_res[i - delay:i].sum() >= 1:
                    test_res[i] = 0

            res_log.append(test_res)
            labels_log.append(test_labels)

        else:
            for k in all_test_data:
                train_data = all_train_data[k]
                train_labels = all_train_labels[k]
                train_timestamps = all_train_timestamps[k]

                test_data = all_test_data[k]
                test_labels = all_test_labels[k]
                test_timestamps = all_test_timestamps[k]

                train_err = self.anomaly_score(model, train_data).detach().cpu().numpy()
                test_err = self.anomaly_score(model, test_data).detach().cpu().numpy()

                ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
                train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
                test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
                train_err_adj = train_err_adj[22:]

                thr = np.mean(train_err_adj) + 4 * np.std(train_err_adj)
                test_res = (test_err_adj > thr) * 1
                res_log_socres.append(test_err_adj)

                for i in range(len(test_res)):
                    if i >= delay and test_res[i-delay:i].sum() >= 1:
                        test_res[i] = 0

                res_log.append(test_res)
                labels_log.append(test_labels)
                timestamps_log.append(test_timestamps)
        t = time.time() - t

        if is_multi:
            if ucr_index == 79 or ucr_index == 108 or ucr_index == 187 or ucr_index == 203:
                labels = np.asarray(labels_log, np.int64)[0]
                pred = np.asarray(res_log, np.int64)[0]

                labels, pred = adjustment(labels, pred)

                eval_res = {
                    'f1': f1_score(labels, pred),
                    'precision': precision_score(labels, pred),
                    'recall': recall_score(labels, pred),
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


                labels = np.asarray(labels_log, np.int64)[0]
                pred = np.asarray(res_log, np.int64)[0]
                # print("labels.shape = ", labels.shape, labels[:5])
                # print("pred.shape = ", pred.shape, pred[:5])

                events_pred = convert_vector_to_events(pred)
                events_gt = convert_vector_to_events(labels)

                Trange = (0, len(labels))
                affiliation = pr_from_events(events_pred, events_gt, Trange)
                vus_results = get_range_vus_roc(labels, pred, 100)  # default slidingWindow = 100

                pred_scores = np.asarray(res_log_socres, np.float64)[0]
                results_f1_pa_k_10 = evaluate.evaluate(pred_scores, labels, k=10)
                results_f1_pa_k_50 = evaluate.evaluate(pred_scores, labels, k=50)
                results_f1_pa_k_90 = evaluate.evaluate(pred_scores, labels, k=90)

                labels, pred = adjustment(labels, pred)

                eval_res = {
                    'f1': f1_score(labels, pred),
                    'precision': precision_score(labels, pred),
                    'recall': recall_score(labels, pred),
                    "Affiliation precision": affiliation['precision'],
                    "Affiliation recall": affiliation['recall'],
                    "R_AUC_ROC": vus_results["R_AUC_ROC"],
                    "R_AUC_PR": vus_results["R_AUC_PR"],
                    "VUS_ROC": vus_results["VUS_ROC"],
                    "VUS_PR": vus_results["VUS_PR"],
                    'f1_pa_10': results_f1_pa_k_10['best_f1_w_pa'],
                    'f1_pa_50': results_f1_pa_k_50['best_f1_w_pa'],
                    'f1_pa_90': results_f1_pa_k_90['best_f1_w_pa'],
                }
        else:

            eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay, pred_scores=res_log_socres)
        eval_res['infer_time'] = t
        return res_log, eval_res

    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)