import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from gpt4ts.models.embed import DataEmbedding


class gpt4ts(nn.Module):

    def __init__(self, max_seq_len, num_classes, var_len, d_model=768, patch_size=8, stride=8, dropout=0.1):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = max_seq_len
        self.max_len = max_seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.gpt_layers = 6
        self.feat_dim = var_len
        self.num_classes = num_classes
        self.d_model = d_model

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, d_model, dropout)

        self.gpt2 = GPT2Model.from_pretrained('/SSD/lz/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        self.gpt2 =  self.gpt2.apply(self.gpt2._init_weights)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
                # param.requires_grad = False
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))
        self.gpt2.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        # self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)

        self.ln_proj = nn.LayerNorm(d_model * self.patch_num)
        self.out_layer = nn.Linear(d_model * self.patch_num, self.num_classes)

    def forward(self, x_enc, x_mark_enc=None):
        x_enc = x_enc.permute(0,2,1)
        B, L, M = x_enc.shape

        # print("x_enc.shape = ", x_enc.shape, B, L, M)

        input_x = rearrange(x_enc, 'b l m -> b m l')
        # print("input_x.shape = ", input_x.shape)
        input_x = self.padding_patch_layer(input_x)
        # print("patch1 input_x.shape = ", input_x.shape)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # print("patch2 input_x.shape = ", input_x.shape)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        # print("patch3 input_x.shape = ", input_x.shape)
        outputs = self.enc_embedding(input_x, None)
        # print("patch4 embd input_x.shape = ", outputs.shape)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        # print("patch5 gpt2 embd input_x.shape = ", outputs.shape)
        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs

