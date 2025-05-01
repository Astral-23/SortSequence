import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.SequenceMaker import SSD  
from model.MLP_ReLU import MLP_ReLU_Net
from model.trainsformer_MadeByChatGpt import TransformerSortNet
from util import train_model
from util import excute_model


EPOCHS = 30
BATCH_SIZE = 16
MIN_VALUE = 0
MAX_VALUE = 1000
NORMALIZE_VALUE = MAX_VALUE
SEED = 42
SEQ_LEN = 10
TRAIN_SIZE = 5000
VAL_SIZE = 1000


# データ準備
train_ds = SSD(TRAIN_SIZE, SEQ_LEN, (MIN_VALUE, MAX_VALUE), SEED, NORMALIZE_VALUE)
val_ds = SSD(VAL_SIZE, SEQ_LEN, (MIN_VALUE, MAX_VALUE), SEED+1, NORMALIZE_VALUE)
test_ds = SSD(VAL_SIZE, SEQ_LEN, (MIN_VALUE, MAX_VALUE), SEED+2, NORMALIZE_VALUE)

dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dl_val = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# 初期化・学習
torch.manual_seed(SEED)
model = TransformerSortNet(SEQ_LEN)
model2 = MLP_ReLU_Net(SEQ_LEN)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

model = train_model(model, criterion, optimizer, EPOCHS, dl_train, dl_val)
model2 = train_model(model2, criterion, optimizer2, EPOCHS, dl_train, dl_val)

excute_model(model, test_ds, NORMALIZE_VALUE, 5)
excute_model(model2, test_ds, NORMALIZE_VALUE, 5)

