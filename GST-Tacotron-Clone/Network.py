import torch.nn as nn
import torch.nn.functional as F
from Modules import *
from GST import GST
from Hyperparameters import Hyperparameters as hp

class Tacotron(nn.Module):
    '''
    TODO: update docstring
    '''
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(hp.vocab), hp.E)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.gst = GST()
    
    def forward(self, texts, mels, ref_mels):
        embedded = self.embedding(texts)
        memory, encoder_hidden = self.encoder(embedded)
        
        style_embed = self.gst(ref_mels)
        style_embed = style_embed.expand_as(memory)
        memory = memory + style_embed
        
        mels_hat, mags_hat, attn_weights = self.decoder(mels, memory)
        return mels_hat, mags_hat, attn_weights

class Encoder(nn.Module):
    '''
    TODO: update docstring
    '''
    
    def __init__(self):
        super().__init__()
        self.prenet = PreNet(in_features=hp.E)
        
        self.conv1d_bank = Conv1dBank(K=hp.K, in_channels=hp.E // 2, out_channels=hp.E // 2)
        
        self.conv1d_1 = Conv1d(in_channels=hp.K * hp.E // 2, out_channels=hp.E // 2, kernel_size=3)
        self.conv1d_2 = Conv1d(in_channels=hp.E // 2, out_channels=hp.E // 2, kernel_size=3)
        self.bn1 = BatchNorm1d(num_features=hp.E // 2)
        self.bn2 = BatchNorm1d(num_features=hp.E // 2)
        
        self.highways = nn.ModuleList()
        for i in range(hp.num_highways):
            self.highways.append(Highway(in_features=hp.E // 2, out_features=hp.E // 2))
        
        self.gru = nn.GRU(input_size=hp.E // 2, hidden_size=hp.E // 2, num_layers=2, bidirectional=True, batch_first=True)
    
    def forward(self, inputs, prev_hidden=None):
        # prenet
        inputs = self.prenet(inputs)
        
        # conv1d projections
        outputs = self.conv1d_1(outputs)
        outputs = self.bn1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv1d_2(outputs)
        outputs = self.bn2(outputs)
        
        outputs = outputs + inputs
        
        # highway
        for layer in self.highways:
            outputs = layer(outputs)
        
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)
        
        return outputs, hidden

class Decoder(nn.Module):
    '''
    TODO: update docstring
    '''
    
    def __init__(self):
        super().__init__()
        self.prenet = PreNet(hp.n_mels)
        self.attn_rnn = AttentionRNN()
        self.attn_projection = nn.Linear(in_features=2 * hp.E, out_features=hp.E)
        self.gru1 = nn.GRU(input_size=hp.E, hidden_size=hp.E, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(input_size=hp.E, hidden_size=hp.E, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(in_features=hp.E, out_features=hp.n_mels*hp.r)
        self.cbhg = DecoderCBHG()
        self.fc2 = nn.Linear(in_features=hp.E, out_features=1 + hp.n_fft // 2)
    
    def forward(self, inputs, memory):
        if self.training:
            # prenet
            outputs = self.prenet(inputs)
            
            attn_weights, outputs, attn_hidden = self.attn_rnn(outputs, memory)
            attn_apply = torch.bmm(attn_weights, memory)
            attn_project = self.attn_projection(torch.cat([attn_apply, outputs], dim=2))
            
            # GRU1
            self.gru1.flatten_parameters()
            outputs1, gru1_hidden = self.gru1(attn_project)
            gru_outputs1 = outputs1 + attn_project
            
            # GRU2
            self.gru2.flatten_parameters()
            outputs2, gru2_hidden = self.gru2(gru_outputs1)
            gru_outputs2 = outputs2 + gru_outputs1

            # generate log melspectrogram
            mels = self.fc1(gru_outputs2)
            
            # CBHG
            out, cbhg_hidden = self.cbhg(mels)
            
            # generate linear spectrogram
            mags = self.fc2(out)
            
            return mels, mags, attn_weights
        else:
            attn_hidden = None
            gru1_hidden = None
            gru2_hidden = None
            
            mels = []
            mags = []
            attn_weights = []
            
            for i in range(hp.max_Ty):
                inputs = self.prenet(inputs)
                attn_weight, outputs, attn_hidden = self.attn_rnn(inputs, memory, attn_hidden)
                attn_weights.append(attn_weight)
                attn_apply = torch.bmm(attn_weight, memory)
                attn_project = self.attn_projection(torch.cat([attn_apply, outputs], dim=-1))
                
                # GRU1
                self.gru1.flatten_parameters()
                outputs1, gru1_hidden = self.gru1(attn_project, gru1_hidden)
                outputs1 = outputs1 + attn_project
                # GRU2
                self.gru2.flatten_parameters()
                outputs2, gru2_hidden = self.gru2(outputs1, gru2_hidden)
                outputs2 = outputs2 + outputs1
                
                # generate log melspectrogram
                mel = self.fc1(outputs2)
                inputs = mel[:, :, -hp.n_mels:]
                mels.append(mel)
                
            mels = torch.cat(mels, dim=1)
            attn_weights = torch.cat(attn_weights, dim=1)
            
            out, cbhg_hidden = self.cbhg(mels)
            mags = self.fc2(out)
            
            return mels, mags, attn_weights
        
class DecoderCBHG(nn.Module):
    '''
    TODO: update docstring
    '''
    
    def __init__(self):
        super().__init__()
        
        self.conv1d_bank = Conv1dBank(K=hp.decoder_K, in_channels=hp.n_mels, out_channels=hp.E // 2)
        
        self.conv1d_1 = Conv1d(in_channels=hp.decoder_K * hp.E // 2, out_channels=hp.E, kernel_size=3)
        self.bn1 = BatchNorm1d(hp.E)
        self.conv1d_2 = Conv1d(in_channels=hp.E, out_channels=hp.n_mels, kernel_size=3)
        self.bn2 = BatchNorm1d(hp.n_mels)
        
        self.highways = nn.ModuleList()
        for i in range(hp.num_highways):
            self.highways.append(Highway(in_features=hp.n_mels, out_features=hp.n_mels))
        
        self.gru = nn.GRU(input_size=hp.n_mels, hidden_size=hp.E // 2, num_layers=2, bidirectional=True, batch_first=True)
    
    def forward(self, inputs, prev_hidden=None):
        inputs = inputs.view(inputs.size(0), -1, hp.n_mels)
        
        # conv1d bank
        outputs = self.conv1d_bank(inputs)
        outputs = max_pool1d(outputs, kernel_size=2)
        
        # conv1d projections
        outputs = self.conv1d_1(outputs)
        outputs = self.bn1(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = self.conv1d_2(outputs)
        outputs = self.bn2(outputs)
        
        outputs = outputs + inputs
        
        # highway net
        for layer in self.highways:
            outputs = layer(outputs)
        
        # bidirection gru
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)
        
        return outputs, hidden
        