import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelName(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size

    def forward(*x):
        pass
 

