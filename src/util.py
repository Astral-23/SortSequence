import torch
import copy

def train_net(model, crit, opt, epochs, dl_train, dl_val):
    best_w, best_loss = None, 1e9
    for ep in range(epochs):
        print(f"Epoch {ep}/{epochs-1}")
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
