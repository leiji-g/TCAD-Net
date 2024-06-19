# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:17:43 2022

@author: ZhengHejie
"""

import torch
import torch.nn as nn
from drop import DropPath
from CSB import CS_Branch

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, device):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.device = device

    def forward(self, x):
        N = x.shape[0]
        x = torch.reshape(x, [N, -1, self.patch_size]).to(self.device)
        return x
        
class SelfAttention(nn.Module):
    def __init__(self, patch_size, heads):
        super(SelfAttention,self).__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = patch_size // heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, patch_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values) 
        keys = self.keys(keys)  
        queries = self.queries(query)
        # 对向量、矩阵、张量的求和运算
        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        
        attention = torch.softmax(energy / (self.patch_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(
            N, query_len, self.heads*self.head_dim)
        
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer块
    :param patch_size: 维度
    :param heads: 多头注意力数
    :param dropout:
    :param forward_expansion: 隐藏层维数
    """

    def __init__(self, patch_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(patch_size, heads)
        self.norm1 = nn.LayerNorm(patch_size)
        self.norm2 = nn.LayerNorm(patch_size)

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.drop = DropPath(0.02)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        x = self.norm1(self.drop(attention) + query)
        out = self.norm2(self.conv(x) + x)
        return out

class Encoder(nn.Module):
    def __init__(self,
                 block_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 num_block):
        super(Encoder, self).__init__()

        self.block_size = block_size
        self.device = device
        self.patch_embedding = PatchEmbedding(patch_size=block_size, device=device)
        self.position_embedding = nn.Embedding(num_block, embedding_dim=block_size)
        self.num_block = num_block

        self.layers = nn.ModuleList(
            [TransformerBlock(block_size,
                              heads,
                              dropout=dropout,
                              forward_expansion=forward_expansion
                              )
             for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = torch.nn.Sequential(nn.Flatten(),
                                       nn.Linear((int((block_size-1)/2+1) + block_size) * num_block, 80),
                                       # nn.Linear(block_size  * num_block, 80),
                                       # nn.Linear(int((block_size - 1) / 2 + 1) * num_block, 80),
                                       nn.ReLU(),
                                       nn.Linear(80, 1),
                                       nn.Sigmoid()
                                       )

        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        positions = torch.arange(0, self.num_block).expand(N, self.num_block).to(self.device)
        # 残差
        patch_em_x = self.patch_embedding(x)
        out = patch_em_x + self.position_embedding(positions)
        esbu_x = CS_Branch(in_channels=1, out_channels=16, kernel_size=3, down_sample=True).cuda()(patch_em_x)
        esbu_x = CS_Branch(in_channels=16, out_channels=32, kernel_size=3, down_sample=False).cuda()(esbu_x)
        esbu_x = CS_Branch(in_channels=32, out_channels=64, kernel_size=3, down_sample=False).cuda()(esbu_x)
        esbu_x = CS_Branch(in_channels=64, out_channels=64, kernel_size=3, down_sample=False).cuda()(esbu_x)
        esbu_x = CS_Branch(in_channels=64, out_channels=128, kernel_size=3, down_sample=False).cuda()(esbu_x)
        esbu_x = CS_Branch(in_channels=128, out_channels=64, kernel_size=3, down_sample=False).cuda()(esbu_x)
        esbu_x = CS_Branch(in_channels=64, out_channels=64, kernel_size=3, down_sample=False).cuda()(esbu_x)
        for layer in self.layers:
            out = layer(out, out, out)
        esbu_x = self.bn(esbu_x)
        out = torch.cat((out, esbu_x), dim=2)
        out = self.dropout(self.mlp(out))
        return out

class ADTransformer(nn.Module):
    def __init__(self,
                 block_size, #20
                 num_layers=8,
                 forward_expansion=4,
                 heads=4,
                 dropout=0.1,
                 device="cuda"):
        super(ADTransformer, self).__init__()
        self.num_block = 64
        self.encoder = Encoder(block_size, num_layers, heads, device,
                               forward_expansion, dropout, self.num_block)


    def forward(self, src):
        enc_src = self.encoder(src)
        return enc_src

    def predict(self, x):
        pred = self.forward(x)
        return pred
    def accuracy_predict(self, x, boundry=0.5):
        pred = self.forward(x)
        ans = []
        for t in pred:
            if t < 0.5:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

    @classmethod
    def from_pretrained(cls,
                        fpath) -> None:
        chkp = torch.load(fpath)
        model = cls(**chkp.pop("config"))
        model.eval()
        model.load_state_dict(chkp.pop("weights"))
        return model





class DetectTransformer(nn.Module):
    def __init__(self,
                 block_size,  # 16
                 sample_size,  # 4
                 original_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=4,
                 dropout=0.1,
                 device="cuda"):
        super(ADTransformer, self).__init__()
        self.num_block = original_size // block_size

        # self.lsm = LSM_IniReconNet(sample_size)
        self.encoder = Encoder(block_size, num_layers, heads, device,
                               forward_expansion, dropout, self.num_block)

    def forward(self, src, trg):
        N = src.shape[0]
        src = self.lsm(src).reshape(N, -1)

        enc_src = self.encoder(src)
        out = self.decoder(trg, enc_src).reshape(N, -1)
        return out