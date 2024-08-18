import torch
import torch.nn as nn
import os

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        # self.is_gpt = configs.is_gpt
        # self.patch_size = configs.patch_size
        # self.pretrain = configs.pretrain
        # self.stride = configs.stride

        # self.encoder = nn.Embedding(configs.pred_len, configs.input_size)
        self.tcn = TemporalConvNet(configs.input_size, configs.num_channels, kernel_size=configs.kernel_size, dropout=configs.dropout)
        self.decoder = nn.Linear(configs.input_size, configs.pred_len)

        self.init_weights()

    def init_weights(self):
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        # print("x.shape = ", x.shape)
        # print("self.tcn = ", self.tcn)
        # emb = self.drop(self.encoder(x))
        y = self.tcn(x)
        # print("y.shape = ", y.shape, y[:, :, -1].shape)
        # print("self.decoder = ", self.decoder)
        result = []
        for i in range(y.shape[-1]):
            o = self.decoder(y[:, :, i])
            result.append(o.contiguous().unsqueeze(-1))
        result = torch.cat(result, dim=2)
        # print("result.shape = ", result.shape)
        return result
