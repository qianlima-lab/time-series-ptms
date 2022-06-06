from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Custom_1, Dataset_Custom_2, Dataset_Custom_NoTime, Dataset_Custom_NoTime_1, Dataset_Custom_NoTime_2,Dataset_Syn
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
from lib.dataloader import get_dataloader
from lib.metrics import All_Metrics
import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.chunk_num,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_optimizer_p(self):
        model_optim_p = optim.Adam([self.model.protos_q,self.model.protos_middle,self.model.protos_k], lr=self.args.learning_rate)
        return model_optim_p

    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_loader, scaler,criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            if batch_x.ndim==4:
                batch_x = batch_x.float().squeeze(3).to(self.device)
                batch_y = batch_y.float().squeeze(3)
            else:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
            
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                print('hh')
            else:
                outputs, dtw_loss = self.model(i, batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

            if self.args.real_value:
                batch_y = scaler.inverse_transform(batch_y[:,-self.args.pred_len:,f_dim:].unsqueeze(3).to(self.device))
            outputs = outputs.unsqueeze(3)
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true) 

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        if self.args.traffic_flow:
            train_loader, vali_loader, test_loader, scaler = get_dataloader(self.args, normalizer=self.args.normalizer,tod=self.args.tod, dow=False, weather=False, single=False)
        else:
            train_data, train_loader = self._get_data(flag = 'train')
            vali_data, vali_loader = self._get_data(flag = 'val')
            test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        model_optim_p = self._select_optimizer_p()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            self.model.init_protos(train_loader)
            
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                model_optim_p.zero_grad()
                
                if batch_x.ndim==4:
                    batch_x = batch_x.float().squeeze(3).to(self.device)
                    batch_y = batch_y.float().squeeze(3)
                else:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    print('hh')
                else:
                    outputs, dtw_loss = self.model(i, batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features=='MS' else 0
                    if self.args.real_value:
                        batch_y = scaler.inverse_transform(batch_y[:,-self.args.pred_len:,f_dim:].unsqueeze(3).to(self.device))
                    outputs = outputs.unsqueeze(3)
                    loss = criterion(outputs, batch_y)
                    #loss += dtw_loss
                    train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    print('hh')
                else:
                    loss.backward(retain_graph=True)
                    dtw_loss.backward(retain_graph=True)
                    model_optim.step()
                    model_optim_p.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, scaler, criterion)
            test_loss = self.vali(test_loader, scaler, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            adjust_learning_rate(model_optim_p, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        train_loader, vali_loader, test_loader, scaler = get_dataloader(self.args, normalizer=self.args.normalizer,tod=self.args.tod, dow=False, weather=False, single=False)

        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        #-------------------
        
        self.model.eval()
        
        preds = None
        trues = None
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            if batch_x.ndim==4:
                batch_x = batch_x.float().squeeze(3).to(self.device)
                batch_y = batch_y.float().squeeze(3)
            else:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                print('hh')
            else:
                outputs, dtw_loss = self.model(i, batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features=='MS' else 0

            true = batch_y[:,-self.args.pred_len:,f_dim:].unsqueeze(3).to(self.device)
            pred = outputs.unsqueeze(3)

            if preds is None:
                preds = pred
                trues = true
            else:
                preds = torch.cat((preds,pred))
                trues = torch.cat((trues,true))

        trues = scaler.inverse_transform(trues)
        if self.args.real_value:
            preds = preds
        else:
            preds = scaler.inverse_transform(preds)

        preds = preds.detach().cpu().numpy()
        trues = trues.detach().cpu().numpy()
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results_ETT/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, rmse, mape, _, _ = All_Metrics(preds, trues, self.args.mae_thresh, self.args.mape_thresh)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape*100))
        print('mae:{}, rmse:{}, mape:{}'.format(mae, rmse, mape))

        np.save(folder_path+'metrics.npy', np.array([mae, rmse, mape]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(i, batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            
            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return