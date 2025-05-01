import torch
import copy
from torch.utils.data import Dataset, DataLoader
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


def train_model(model, crit, opt, dl_train, dl_val):
    best_w, best_loss = None, float("inf")
    for ep in range(EPOCHS):
        print(f"Epoch {ep+1}/{EPOCHS}")
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


def excute_model_corner(models):
    corner_cases = [
        {
            "name": "All Zeros",
            "input": torch.zeros(SEQ_LEN, dtype=torch.float32),
        },  # X = [0, 0, ..., 0]
        {
            "name": "All Max Value",
            "input": torch.full((SEQ_LEN,), MAX_VALUE, dtype=torch.float32),
        },  # X = [MAX_VALUE, MAX_VALUE, ..., MAX_VALUE]
        {
            "name": "All Mid Value",
            "input": torch.full((SEQ_LEN,), (MAX_VALUE + MIN_VALUE)/2, dtype=torch.float32),
        },  # X = [MIN_VALUE, MIN_VALUE, ..., MIN_VALUE]
    ]

    for case in corner_cases:
        print(f"\n--- Test with corner case: {case['name']} ---")
        X = case["input"]
        print(f"Input X = {X.numpy().astype(int)}")

        # 各モデルに対して予測
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model((X/NORMALIZE_VALUE).unsqueeze(0)) * NORMALIZE_VALUE  # バッチ化
                p_np = pred.squeeze(0).numpy()  # shape: [output_dim]
                p_formatted = [f"{v:.2f}" for v in p_np]
                print(f"  Model: {type(model).__name__}")
                print(f"    Predicted output: {p_formatted}")


def excute_model(models, test_ds, data_num=None):
    print("\nPredictions (denormalized):")

    if data_num is None:
        data_num = len(test_ds)

    # モデルをリストでない場合でもリストに変換（柔軟性向上）
    if not isinstance(models, (list, tuple)):
        models = [models]

    # すべてのモデルを eval モードに
    for model in models:
        model.eval()

    with torch.no_grad():
        for X, y in DataLoader(test_ds, batch_size=data_num, shuffle=False):
            print("-" * 40)
            for xi, yi in zip(X, y):
                xi_denorm = (xi * NORMALIZE_VALUE).numpy()
                yi_denorm = (yi * NORMALIZE_VALUE).numpy()

                print(f"X={xi_denorm.astype(int)}, sorted={yi_denorm.astype(int)}")
                for model in models:
                    model.eval()
                    pred = model(xi.unsqueeze(0)) * NORMALIZE_VALUE
                    p_np = pred.squeeze(0).numpy()
                    p_formatted = [f"{v:.2f}" for v in p_np]
                    print(f"  model: {type(model).__name__}")
                    print(f"    pred: {p_formatted}")
                print()
            break
