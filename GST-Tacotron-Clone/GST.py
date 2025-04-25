import torch.nn as nn
from ReferenceEncoder import ReferenceEncoder
from StyleTokenLayer import StyleLokenLayer

class GST(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = ReferenceEncoder()
        self.stl = StyleLokenLayer()

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)
        return style_embed