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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.SequenceMaker import RandomGenerator
from model.MLP_ReLU import MLP_ReLU_Net
from model.trainsformer_MadeByChatGpt import TransformerSortNet
from util import train_model, excute_model_corner, excute_model
from params import (
    EPOCHS,
    BATCH_SIZE,
    MIN_VALUE,
    MAX_VALUE,
    NORMALIZE_VALUE,
    SEED,
    SEQ_LEN,
    TRAIN_SIZE,
    VAL_SIZE,
)


# データ準備
train_ds = RandomGenerator(
    TRAIN_SIZE, SEQ_LEN, (MIN_VALUE, MAX_VALUE), SEED, NORMALIZE_VALUE
)
val_ds = RandomGenerator(
    VAL_SIZE, SEQ_LEN, (MIN_VALUE, MAX_VALUE), SEED + 1, NORMALIZE_VALUE
)
test_ds = RandomGenerator(
    VAL_SIZE, SEQ_LEN, (MIN_VALUE, MAX_VALUE), SEED + 2, NORMALIZE_VALUE
)

dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dl_val = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# 初期化・学習
torch.manual_seed(SEED)
model = TransformerSortNet(SEQ_LEN)
model2 = MLP_ReLU_Net(SEQ_LEN)
model3 = TransformerSortNet(SEQ_LEN)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=0.01)
optimizer3 = optim.AdamW(model3.parameters(), lr=1e-3, weight_decay=0.01)


model = train_model(model, criterion, optimizer, dl_train, dl_val)
model2 = train_model(model2, criterion, optimizer2, dl_train, dl_val)
#model3 = train_model(model3, criterion, optimizer3, dl_train, dl_val)


excute_model((model, model2), test_ds, 5)
excute_model_corner((model, model2))
