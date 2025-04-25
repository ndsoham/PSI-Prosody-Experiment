import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from Hyperparameters import Hyperparameters as hp

class StyleLokenLayer(nn.Module):
    '''
    TODO: update docstring
    '''
    def __init__(self):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num,
                                                    hp.E // hp.num_heads))
        d_q = hp.E // 2
        d_k = hp.E // hp.num_heads
        self.attention = MultiHeadAttention(query_dim=d_q,
                                            key_dim=d_k,
                                            num_units=hp.E,
                                            num_heads=hp.num_heads)
        init.normal_(self.embed, mean=0, std=0.5)
    
    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        style_embed = self.attention(query, keys)
        return style_embed

class MultiHeadAttention(nn.Module):
    '''
    TODO: update docstring
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
    
    def forward(self, query, key):
        querys = self.W_query(query)
        keys = self.W_key(key)
        values = self.W_value(key)
        
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)
        
        scores = torch.matmul(querys, keys.transpose(2, 3))
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)
        
        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)
        
        return out
    
    
    