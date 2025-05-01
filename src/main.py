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
from model.MLP_ReLU import Net
from model.trainsformer_MadeByChatGpt import TransformerSortNet
from util import train_net

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
dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dl_val = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# 初期化・学習
torch.manual_seed(SEED)
model = TransformerSortNet(SEQ_LEN)
model2 = Net(SEQ_LEN)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

model = train_net(model, criterion, optimizer, EPOCHS, dl_train, dl_val)
model2 = train_net(model2, criterion, optimizer2, EPOCHS, dl_train, dl_val)

# サンプル表示
print("\nSample Predictions (denormalized):")
test_ds =  SSD(100, SEQ_LEN, (MIN_VALUE, MAX_VALUE), SEED+2, NORMALIZE_VALUE)
model.eval()
model2.eval()
with torch.no_grad():
    for X, y in DataLoader(test_ds, batch_size=5):
        pred1 = model(X) * MAX_VALUE
        pred2 = model2(X) * MAX_VALUE  # ← 追加

        X_np = (X.numpy() * MAX_VALUE).astype(int)
        y_np = (y.numpy() * MAX_VALUE).astype(int)
        p1_np = pred1.numpy()
        p2_np = pred2.numpy()  # ← 追加

        for xi, yi, pi1, pi2 in zip(X_np, y_np, p1_np, p2_np):
            pi1_formatted = [f"{v:.2f}" for v in pi1]
            pi2_formatted = [f"{v:.2f}" for v in pi2]
            print(f"X={xi}, sorted={yi}")
            print(f"  TransformerSortNet pred: {pi1_formatted}")
            print(f"  Net (MLP)             pred: {pi2_formatted}")
        break