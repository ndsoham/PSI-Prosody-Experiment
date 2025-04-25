import torch
from torch import nn
class ReferenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stck = nn.Sequential(
            nn.Conv2d()
        )
        