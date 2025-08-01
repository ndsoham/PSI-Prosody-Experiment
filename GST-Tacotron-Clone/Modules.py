import torch
import torch.nn as nn
import torch.nn.functional as F
from Hyperparameters import Hyperparameters as hp

class Conv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        '''
        TODO: update docstrings
        '''
        super().__init__()
        if padding == 'same':
            left = (kernel_size - 1) // 2
            right = (kernel_size - 1) - left
            self.pad = (left, right)
        else:
            self.pad = (0, 0)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)
        inputs = F.pad(inputs, self.pad)
        out = self.conv1d(inputs)
        out = torch.transpose(out, 1, 2)
        return out

class Highway(nn.Module):
    
    def __init__(self, in_features, out_features):
        '''
        TODO: update docstring
        '''
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        
    def forward(self, inputs):
        H = self.linear1(inputs)
        H = F.relu(H)
        T = self.linear2(inputs)
        T = F.sigmoid(T)
        
        out = H * T + inputs * (1.0 - T)
        return out
    
class Conv1dBank(nn.Module):
    '''
    TODO: update docstring
    '''
    
    def __init__(self, K, in_channels, out_channels):
        super().__init__()
        self.bank = nn.ModuleList()
        for k in range(1, K+1):
            self.bank.append(Conv1d(in_channels, out_channels, kernel_size=k))
        self.bn = BatchNorm1d(out_channels*K)
    
    def forward(self, inputs):
        outputs = self.bank[0](inputs)
        for k in range(1, len(self.bank)):
            output = self.bank[k](inputs)
            outputs = torch.cat([outputs, output], dim=2)
        outputs = self.bn(outputs)
        outputs = F.relu(outputs)
        
        return outputs

class BatchNorm1d(nn.Module):
    '''
    TODO: update docstring
    '''
    
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
    
    def forward(self, inputs):
        out = self.bn(inputs.transpose(1, 2).contiguous())
        return out.transpose(1, 2)
    
class PreNet(nn.Module):
    '''
    TODO: update docstring
    '''
    
    def __init__(self, in_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hp.E)
        self.linear2 = nn.Linear(hp.E, hp.E // 2)
        self.dropout1 = nn.Dropout(hp.dropout_p)
        self.dropout2 = nn.Dropout(hp.dropout_p)
    
    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = F.relu(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.linear2(outputs)
        outputs = F.relu(outputs)
        outputs = self.dropout2(outputs)
        return outputs


class AttentionRNN(nn.Module):
    '''
    TODO: update docstring
    '''
    
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=hp.E // 2, hidden_size=hp.E, batch_first=True, bidirectional=False)
        self.W = nn.Linear(in_features=hp.E, out_features=hp.E, bias=False)
        self.U = nn.Linear(in_features=hp.E, out_features=hp.E, bias=False)
        self.v = nn.Linear(in_features=hp.E, out_features=1, bias=False)
    
    def forward(self, inputs, memory, prev_hidden=None):
        T_x = memory.size(1)
        T_y = inputs.size(1)
        
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(inputs, prev_hidden)
        w = self.W(outputs).unsqueeze(2).expand(-1, -1, T_x, -1)
        u = self.U(memory).unsqueeze(1).expand(-1, T_y, -1, -1)
        attn_weights = self.v(F.tanh(w + u).view(-1, hp.E)).view(-1, T_y, T_x)
        attn_weights = F.softmax(attn_weights, 2)
        
        return attn_weights, outputs, hidden

def max_pool1d(inputs, kernel_size, stride=1, padding='same'):
    '''
    TODO: update docstring
    '''
    inputs = inputs.transpose(1, 2)
    if padding == 'same':
        left = (kernel_size - 1) // 2
        right = (kernel_size - 1) - left
        pad = (left, right)
    else:
        pad = (0 , 0)
    inputs = F.pad(inputs, pad)
    outputs = F.max_pool1d(inputs, kernel_size, stride)
    outputs = outputs.transpose(1, 2)
    
    return outputs

