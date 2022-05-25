import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from utils import data_slice,split_N_pad
import time
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

class AnomalyAttention(nn.Module):
    def __init__(self, N, d_model):
        super(AnomalyAttention, self).__init__()
        self.d_model = d_model
        self.N = N

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Ws = nn.Linear(d_model, 1, bias=False)
        self.Q = self.K = self.V = self.sigma = torch.zeros((N, d_model))
        self.P = torch.zeros((N, N))
        self.S = torch.zeros((N, N))

    def forward(self, x):
        #x :[batch,N,d_model]
        self.initialize(x)
        self.S = self.series_association()
        self.P = self.prior_association()
        Z = self.reconstruction()
        return Z

    def initialize(self, x):
        self.Q = self.Wq(x)
        self.K = self.Wk(x)
        self.V = self.Wv(x)
        self.sigma = self.Ws(x)

    @staticmethod
    def gaussian_kernel(mean, sigma):
        normalize = 1 / (math.sqrt(2 * torch.pi) * torch.abs(sigma))
        return normalize * torch.exp(-0.5 * (mean / sigma).pow(2))

    def prior_association(self):
        # qwe = torch.from_numpy(
        #     np.abs(np.indices((self.N, self.N))[0] - np.indices((self.N, self.N))[1])
        # ).cuda
        qwe = torch.from_numpy(
            np.abs(np.indices((self.N, self.N))[0] - np.indices((self.N, self.N))[1])
        )
        if torch.cuda.is_available():
            qwe = qwe.cuda()
        #原 gaussian: [batch,N,N]
        #因为是高斯所以这里行列求和都一样
        gaussian = self.gaussian_kernel(qwe.double(), self.sigma)
        gaussian /= gaussian.sum(dim=-1).view(-1,self.N,1)
        return gaussian

    def series_association(self):
        # 原 [N,N]
        # return F.softmax(self.Q @ self.K.T / math.sqrt(self.d_model), dim=0)
        # 现 [batch,N,N],是列方向的softmax？,应该是不对的，得改成行方向的softmax，根据下游的reconstruction来看
        return F.softmax(torch.matmul(self.Q,self.K.transpose(1,2)) / math.sqrt(self.d_model), dim=2)

    def reconstruction(self):
        return torch.matmul(self.S,self.V)

class AnomalyTransformerBlock(nn.Module):
    def __init__(self, N, d_model):
        super().__init__()
        self.N, self.d_model = N, d_model

        self.attention = AnomalyAttention(self.N, self.d_model)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        # x: [batch,N,d_model]
        x_identity = x
        x = self.attention(x)
        z = self.ln1(x + x_identity)
        z_identity = z
        z = self.ff(z)
        z = self.ln2(z + z_identity)
        
        # z: [batch,N,d_model]
        return z

class AnomalyTransformer(nn.Module):
    def __init__(self,batch_size, N, in_channel, d_model, layers, lambda_):
        super().__init__()
        self.batch_size = batch_size
        self.in_channel = in_channel
        self.N = N
        self.d_model = d_model

        self.input2hidden = nn.Linear(self.in_channel,self.d_model)
        self.hidden2output = nn.Linear(self.d_model,self.in_channel)
        self.blocks = nn.ModuleList(
            [AnomalyTransformerBlock(self.N, self.d_model) for _ in range(layers)]
        )
        self.output = None
        self.lambda_ = lambda_

        self.P_layers = []
        self.S_layers = []
    def to_string(self):
        return 'in_channel:%d_N:%d_dmodel:%d_' % (self.in_channel,self.N,self.d_model)

    def forward(self, x):
        
        # x: [batch,N,in_channel]
        self.P_layers = []
        self.S_layers = []
        x = self.input2hidden(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            # x: [batch,N,d_model]
            self.P_layers.append(block.attention.P)
            self.S_layers.append(block.attention.S)
        self.output = self.hidden2output(x)
        # output: [batch,N,in_channel]
        return self.output
    
    # def layer_association_discrepancy(self, Pl, Sl, x):
    #     rowwise_kl = lambda row: (
    #         F.kl_div(Pl[row, :], Sl[row, :]) + F.kl_div(Sl[row, :], Pl[row, :])
    #     )
    #     ad_vector = torch.concat(
    #         [rowwise_kl(row).unsqueeze(0) for row in range(Pl.shape[0])]
    #     )
    #     return ad_vector
    # ad_vector: [N]
    
    # def rowwise_kl (self,Pl,Sl,idx,row):
    #     return F.kl_div(Pl[idx,row, :], Sl[idx,row, :]) + F.kl_div(Sl[idx,row, :], Pl[idx,row, :])
    # def layer_association_discrepancy(self, Pl, Sl, x):
        
    #     wholetmp=[]
    #     for idx in range(Pl.shape[0]):
    #         rowtmp=[]
    #         for row in range(Pl.shape[1]):
    #             rowtmp.append(self.rowwise_kl(Pl,Sl,idx,row).unsqueeze(0))
    #         wholetmp.append(torch.cat(rowtmp))
                
    #     ad_vector = torch.cat( 
    #         wholetmp
    #     ).reshape([-1,Pl.shape[1]])
    #     #ad_vector: [batch,N]
    #     return ad_vector
    
    def rowwise_kl(self, row, Pl, Sl, eps=1e-4):
        Pl_r = Pl[:,row,:]
        Sl_r = Sl[:,row,:]
        Pl_r = (Pl_r+ eps) / torch.sum(Pl_r + eps, dim=-1, keepdims=True)
        Sl_r = (Sl_r + eps) / torch.sum(Sl_r+ eps, dim=-1, keepdims=True)
        '''TODO:改这个函数'''
        ret = torch.sum( 
            F.kl_div( torch.log(Pl_r), Sl_r, reduction='none') + F.kl_div( torch.log(Sl_r), Pl_r, reduction='none'), dim=1
         )
        return ret
    def layer_association_discrepancy(self, Pl, Sl, x):
        ad_vector = torch.concat(
            [self.rowwise_kl(row, Pl, Sl).unsqueeze(1) for row in range(Pl.shape[1])], dim=1
        )
        return ad_vector

    def association_discrepancy(self, P_list, S_list, x):
        
        ret = (1 / len(P_list)) * sum(
            [
                self.layer_association_discrepancy(P, S, x)
                for P, S in zip(P_list, S_list)
            ]
        )
        # ret: [batch,N]
        return ret

    def loss_function(self, x_hat, P_list, S_list, lambda_, x):
        #P_list: [layers,batch,N,N]
        #S_list: [layers,batch,N,N]
        frob_norm = torch.linalg.matrix_norm(x_hat - x, ord="fro")
        ret = frob_norm - (
            lambda_
            * torch.linalg.norm(self.association_discrepancy(P_list, S_list, x),dim=1, ord=1)
        )
        return ret.mean()

    def min_loss(self, x):
        
        P_list = self.P_layers
        S_list = [S.detach() for S in self.S_layers]
        # S_list = self.S_layers
        lambda_ = -self.lambda_
        return self.loss_function(self.output, P_list, S_list, lambda_, x)

    def max_loss(self, x):
        P_list = [P.detach() for P in self.P_layers]
        # P_list = self.P_layers
        S_list = self.S_layers
        lambda_ = self.lambda_
        return self.loss_function(self.output, P_list, S_list, lambda_, x)
    
    def anomaly_score_whole(self, x):
        # x:[length,dim]
        x = np.array(split_N_pad(x.reshape([-1,1]),self.N))
        '''TODO:测试data_slice'''
        data = torch.from_numpy(x)
        if torch.cuda.is_available():
            data = data.cuda()
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=False, drop_last=False)
        scores=[]
        for step, batch in enumerate(dataloader):
            batch=batch[0]
            score = self.anomaly_score(batch)
            scores.append(score)
        return torch.cat(scores).flatten()
            
    

    def anomaly_score(self, x):
        # 原 x:[N,in_channel]
        output = self.forward(x)
        tmp = -self.association_discrepancy(self.P_layers, self.S_layers, x)
        ad = F.softmax(
            tmp, dim=0
        )
        assert ad.shape[1] == self.N

        # norm = torch.tensor(
        #     [
        #         torch.linalg.norm(x[i, :] - self.output[i, :], ord=2)
        #         for i in range(self.N)
        #     ]
        # )
        norm = []
        for idx in range(x.shape[0]):
            tmp = torch.tensor(
                [
                    torch.linalg.norm(x[idx,i, :] - self.output[idx,i, :], ord=2)
                    for i in range(self.N)
                ]
            )
            norm.append(tmp)
        norm = torch.cat(norm).reshape([-1,self.N])
        assert norm.shape[1] == self.N
        score = torch.mul(ad, norm)
        return score
