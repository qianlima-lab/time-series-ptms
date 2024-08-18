"""
This code is based on huggingface,
https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py

MIT License

Copyright (c) 2018 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OFS CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
# Arxiv Link https://arxiv.org/pdf/1907.00235.pdf


import numpy as np
import torch
import torch.nn as nn
import math
import copy
from torch.nn.parameter import Parameter
from torch.autograd import Function


def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)



def _entmax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for 1.5-entmax: optimal threshold and support size.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.
    dim : int
        The dimension along which to apply 1.5-entmax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star, support_size


class Entmax15Function(Function):
    @classmethod
    def forward(cls, ctx, X: torch.Tensor, dim=0, k=None):
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


def entmax15(X, dim=-1, k=None):
    """1.5-entmax: normalizing sparse transform (a la softmax).
    Solves the optimization problem:
        max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.
    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    dim : int
        The dimension along which to apply 1.5-entmax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """

    return Entmax15Function.apply(X, dim, k)


def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for sparsemax: optimal threshold and support size.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.
    dim : int
        The dimension along which to apply sparsemax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


class SparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


def sparsemax(X, dim=-1, k=None):
    """sparsemax: normalizing sparse transform (a la softmax).
    Solves the projection:
        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    dim : int
        The dimension along which to apply sparsemax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """

    return SparsemaxFunction.apply(X, dim, k)


class Sparsemax(nn.Module):
    def __init__(self, dim=-1, k=None):
        """sparsemax: normalizing sparse transform (a la softmax).
        Solves the projection:
            min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.
        Parameters
        ----------
        dim : int
            The dimension along which to apply sparsemax.
        k : int or None
            number of largest elements to partial-sort over. For optimal
            performance, should be slightly bigger than the expected number of
            nonzeros in the solution. If the solution is more than k-sparse,
            this function is recursively called with a 2*k schedule.
            If `None`, full sorting is performed from the beginning.
        """
        self.dim = dim
        self.k = k
        super(Sparsemax, self).__init__()

    def forward(self, X):
        return sparsemax(X, dim=self.dim, k=self.k)


def swish(x):
    return x * torch.sigmoid(x)


activation_dict = {"ReLU": torch.nn.ReLU(), "Softplus": torch.nn.Softplus(), "Swish": swish,
                   "entmax": entmax15, "sparsemax": sparsemax, "Softmax": torch.nn.Softmax}



def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)

ACT_FNS = {
    'relu': nn.ReLU(),
    'swish': swish,
    'gelu': gelu
}


class Attention(nn.Module):
    def __init__(self, n_head, n_embd, win_len, scale, q_len, sub_len, sparse=True, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Attention, self).__init__()

        if(sparse):
            print('Activate log sparse!')
            mask = self.log_mask(win_len, sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        self.register_buffer('mask_tri', mask)
        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while(index >= 0):
                if((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor, activation="Softmax"):
        activation = activation_dict[activation](dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):

        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn


class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            # print("test.shape = ", self.b.shape, x.view(-1, x.size(-1)).shape, self.w.shape, x.shape)
            # if x.size(-1) > self.w.shape[0]:
            #     # min_len = min(x.size(-1), self.w.shape[0])
            #     self.w = torch.nn.functional.pad(self.w, (0, 0, 0, (x.size(-1)-self.w.shape[0])), "constant", 0)
            #     self.w = torch.nn.Parameter(self.w)  # 转换为 torch.nn.Parameter
            #     x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            # else:
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_embd, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))
        self.e = e

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.e)
        return self.g * x + self.b


class MLP(nn.Module):
    def __init__(self, n_state, n_embd, acf='relu'):
        super(MLP, self).__init__()
        n_embd = n_embd
        self.c_fc = Conv1D(n_state, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_state)
        self.act = ACT_FNS[acf]
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        hidden1 = self.act(self.c_fc(x))
        hidden2 = self.c_proj(hidden1)
        return self.dropout(hidden2)


class Block(nn.Module):
    def __init__(self, n_head, win_len, n_embd, scale, q_len, sub_len):
        super(Block, self).__init__()
        n_embd = n_embd
        self.attn = Attention(n_head, n_embd, win_len, scale, q_len, sub_len)
        self.ln_1 = LayerNorm(n_embd)
        self.mlp = MLP(4 * n_embd, n_embd)
        self.ln_2 = LayerNorm(n_embd)

    def forward(self, x):
        attn = self.attn(x)
        ln1 = self.ln_1(x + attn)
        mlp = self.mlp(ln1)
        hidden = self.ln_2(ln1 + mlp)
        return hidden


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, n_time_series, n_head, sub_len, num_layer, n_embd,
                 forecast_history: int, dropout: float, scale_att, q_len, seq_num=None):
        super(TransformerModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = n_time_series
        self.n_head = n_head
        self.seq_num = None
        if seq_num:
            self.seq_num = seq_num
            self.id_embed = nn.Embedding(seq_num, n_embd)
            nn.init.normal_(self.id_embed.weight, std=0.02)
        self.n_embd = n_embd
        self.win_len = forecast_history
        # The following is the implementation of this paragraph
        """ For positional encoding in Transformer, we use learnable position embedding.
        For covariates, following [3], we use all or part of year, month, day-of-the-week,
        hour-of-the-day, minute-of-the-hour, age and time-series-ID according to the granularities of datasets.
        age is the distance to the first observation in that time series [3]. Each of them except time series
        ID has only one dimension and is normalized to have zero mean and unit variance (if applicable).
        """
        self.po_embed = nn.Embedding(forecast_history, n_embd)
        self.drop_em = nn.Dropout(dropout)
        block = Block(n_head, forecast_history, n_embd + n_time_series, scale=scale_att,
                      q_len=q_len, sub_len=sub_len)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layer)])
        nn.init.normal_(self.po_embed.weight, std=0.02)

    def forward(self, series_id: int, x: torch.Tensor):
        """Runs  forward pass of the DecoderTransformer model.

        :param series_id:   ID of the time series
        :type series_id: int
        :param x: [description]
        :type x: torch.Tensor
        :return: [description]
        :rtype: [type]
        """
        batch_size = x.size(0)
        length = x.size(1)  # (Batch_size, length, input_dim)
        embedding_sum = torch.zeros(batch_size, length, self.n_embd).to(self.device)
        if self.seq_num:
            embedding_sum = torch.zeros(batch_size, length)
            embedding_sum = embedding_sum.fill_(series_id).type(torch.LongTensor).to(self.device)
            embedding_sum = self.id_embed(embedding_sum)
        #print("shape below")
        #print(embedding_sum.shape)
        #print(x.shape)
        #print(series_id)
        position = torch.tensor(torch.arange(length), dtype=torch.long).to(self.device)
        po_embedding = self.po_embed(position)
        embedding_sum[:] = po_embedding
        x = torch.cat((x, embedding_sum), dim=2)
        for block in self.blocks:
            x = block(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        n_time_series = configs.enc_in
        n_head = configs.n_heads
        num_layer = configs.e_layers
        n_embd = configs.d_model // configs.n_heads
        forecast_history = configs.seq_len
        dropout = configs.dropout
        q_len = configs.seq_len // 6
        activation = configs.activation
        forecast_length = configs.pred_len
        scale_att: bool = False
        seq_num1=None
        sub_len=1
        mu=None
        """
        Args:
            n_time_series: Number of time series present in input
            n_head: Number of heads in the MultiHeadAttention mechanism
            seq_num: The number of targets to forecast
            sub_len: sub_len of the sparse attention
            num_layer: The number of transformer blocks in the model.
            n_embd: The dimention of Position embedding and time series ID embedding
            forecast_history: The number of historical steps fed into the time series model
            dropout: The dropout for the embedding of the model.
            additional_params: Additional parameters used to initalize the attention model. Can inc
        """
        super(Model, self).__init__()
        print("trans para = ", n_time_series, n_head, sub_len, num_layer, n_embd, forecast_history, dropout, scale_att, q_len)
        self.transformer = TransformerModel(n_time_series, n_head, sub_len, num_layer, n_embd, forecast_history, dropout, scale_att, q_len, seq_num=seq_num1)
        self.softplus = nn.Softplus()
        self.mu = torch.nn.Linear(n_time_series + n_embd, n_time_series, bias=True)
        self.sigma = torch.nn.Linear(n_time_series + n_embd, n_time_series, bias=True)
        self._initialize_weights()
        self.mu_mode = mu
        self.forecast_len_layer = None
        if forecast_length:
            self.forecast_len_layer = nn.Linear(forecast_history, forecast_length)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, series_id: int = None):
        """
        Args:
            x: Tensor of dimension (batch_size, seq_len, number_of_time_series)
            series_id: Optional id of the series in the dataframe. Currently  not supported
        Returns:
            Case 1: tensor of dimension (batch_size, forecast_length)
            Case 2: GLoss sigma and mu: tuple of ((batch_size, forecast_history, 1), (batch_size, forecast_history, 1))
        """
        # print("x.shape = ", x.shape)
        # print(" self.transformer = ",  self.transformer)
        h = self.transformer(series_id, x)
        # print("h.shape")
        mu = self.mu(h)
        sigma = self.sigma(h)
        if self.mu_mode:
            sigma = self.softplus(sigma)
            return mu, sigma
        if self.forecast_len_layer:
            # Swap to (batch_size, 1, features) for linear layer
            sigma = sigma.permute(0, 2, 1)
            # Output (batch_size, forecast_len_)
            sigma = self.forecast_len_layer(sigma).permute(0, 2, 1)
        return sigma
