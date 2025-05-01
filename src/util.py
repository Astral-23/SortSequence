import torch
import copy
from torch.utils.data import Dataset, DataLoader

def train_model(model, crit, opt, epochs, dl_train, dl_val):
    best_w, best_loss = None, float('inf')
    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")
        for phase, dl in [("train", dl_train), ("val", dl_val)]:
            model.train() if phase == "train" else model.eval()
            total_loss = 0
            for X, y in dl:
                opt.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    pred = model(X)
                    loss = crit(pred, y)
                    if phase == "train":
                        loss.backward()
                        opt.step()
                total_loss += loss.item() * X.size(0)
            epoch_loss = total_loss / len(dl.dataset)
            print(f"  {phase} Loss: {epoch_loss:.4f}")
            if phase == "val" and epoch_loss < best_loss:
                best_loss, best_w = epoch_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_w)
    return model


def excute_model(model, test_ds,  NORMALIZE_VALUE, data_num=None):
    print("\nPredictions (denormalized):")
    model.eval()
    if data_num is None:
        data_num = len(test_ds)
        
    with torch.no_grad():
        for X, y in DataLoader(test_ds, batch_size=data_num, shuffle=False):
            pred = model(X) * NORMALIZE_VALUE

            X_np = (X.numpy() * NORMALIZE_VALUE).astype(int)
            y_np = (y.numpy() * NORMALIZE_VALUE).astype(int)
            p_np = pred.numpy()

            for xi, yi, pi in zip(X_np, y_np, p_np):
                pi_formatted = [f"{v:.2f}" for v in pi]
                print(f"X={xi}, sorted={yi}")
                print(f"  model: {type(model).__name__} pred: {pi_formatted}")
            break