import torch.nn as nn
import torch.nn.functional as F


class MLP_ReLU_Net(nn.Module):
    def __init__(self, SEQ_LEN):
        super(MLP_ReLU_Net, self).__init__()

        self.fc1 = nn.Linear(SEQ_LEN, SEQ_LEN)
        self.fc2 = nn.Linear(SEQ_LEN, SEQ_LEN)
        self.fc3 = nn.Linear(SEQ_LEN, SEQ_LEN)
        self.fc4 = nn.Linear(SEQ_LEN, SEQ_LEN)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU 活性化関数を追加
        x = F.relu(self.fc2(x))  # ReLU 活性化関数を追加
        x = F.relu(self.fc3(x))  # ReLU 活性化関数を追加
        x = self.fc4(x)  # 最後の層は ReLU を適用しないことが一般的
        return x
