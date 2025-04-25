import torch.nn as nn
import torch.nn.functional as F
from Hyperparameters import Hyperparameters as hp

class ReferenceEncoder(nn.Module):
    '''
    TODO: update docstring
    '''
    
    def __init__(self):
        super().__init__()
        conv_length = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters
        convs = [
            nn.Conv2d(in_channels=filters[i],
                      out_channels=filters[i+1],
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1))
            for i in range(conv_length)
            ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [
                nn.BatchNorm2d(num_features=hp.ref_enc_filters[i])
                for i in range(conv_length)
            ]
        )
        out_channels = self.calculate_channels(hp.n_mels, 3, 2, 1, conv_length)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.E // 2,
                          batch_first=True)
        
    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, hp.n_mels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)
        
        out = out.transpose(1, 2)
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)
        
        self.gru.flatten_parameters()
        memory, out = self.gru(out)
        
        return out.squeeze(0)
            
    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

