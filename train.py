import random
import os
import time
from itertools import count
from datetime import datetime
# 3rd-party Modules
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# self-defined Modules
from config import get_args
from model import ModelName
from datasets import DatasetName
from utils import *

if __name__ == '__main__':
    args = get_args()
    for name, arg in vars(args).items():
        print('%s: %s' % (name, arg))

    # Reproducibility
    set_seed(args.seed)

    use_cuda = (args.gpu >= 0)
    print(f"Using device, {args.gpu}\n")

    train_device = torch.device('cuda:%s' % args.gpu) if use_cuda else torch.device('cpu')

