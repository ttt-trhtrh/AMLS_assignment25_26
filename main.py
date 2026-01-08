
from __future__ import annotations

print("=== main.py is running ===")

import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

# Model A (Classical ML)
from sklearn.linear_model import LogisticRegression  # Logistic Regression（逻辑回归）
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model B (Deep Learning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Dataset
import medmnist
from medmnist import BreastMNIST


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if no CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    y_true, y_pred: shape (N,), values in {0,1}
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ----------------------------
# Data
# ----------------------------
def get_datasets(root: str):
    """
    MedMNIST provides official splits: train/val/test.
    We'll download (if needed) into ./Datasets (ignored by .gitignore).
    """
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),  # (H,W) -> (1,H,W), float in [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5]),  # simple normalization
        ]
    )

    train_ds = BreastMNIST(split="train", root=root, transform=tfm, download=True)
    val_ds = BreastMNIST(split="val", root=root, transform=tfm, download=True)
    test_ds = BreastMNIST(split="test", root=root, transform=tfm, download=True)
    return train_ds, val_ds, test_ds


def dataset_to_numpy_flat(ds) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert dataset to numpy arrays for classical ML:
    X: (N, 784), y: (N,)
    """
    xs = []
    ys = []
    for x, y in ds:
        # x: torch tensor (1,28,28)
        x_np = x.numpy().reshape(-1)  # 784
        y_np = int(y.item())  # label is shape (1,) in MedMNIST
        xs.append(x_np)
        ys.append(y_np)
    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return X, y


# ----------------------------
# Model B: Simple CNN (Convolutional Neural Network, 卷积神经网络)
# ----------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # binary logits
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def eval_cnn(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_true = []
    all_pred = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float().view(-1)  # (B,)
        logits = model(x).view(-1)         # (B,)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).long().cpu().numpy()
        true = y.long().cpu().numpy()
        all_true.append(true)
        all_pred.append(pred)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return compute_metrics(y_true, y_pred)


def train_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> nn.Module:
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # Adam（自适应优化器）
    loss_fn = nn.BCEWithLogitsLoss()  # binary classification loss on logits

    best_val_f1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"[CNN] epoch {ep}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device).float().view(-1, 1)  # (B,1)

            opt.zero_grad()
            logits = model(x)  # (B,1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)

        train_loss = running / len(train_loader.dataset)
        val_metrics = eval_cnn(model, val_loader, device)

        print(
            f"[CNN] epoch {ep}/{epochs} | train_loss={train_loss:.4f} "
            f"| val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ----------------------------
# Main
# ----------------------------
@dataclass
class Config:
    seed: int = 42
    dataset_root: str = "./Datasets"
    batch_size: int = 128
    cnn_epochs: int = 5
    cnn_lr: float = 1e-3
    out_json: str = "results_baseline.json"


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    os.makedirs(cfg.dataset_root, exist_ok=True)

    print("Loading BreastMNIST (train/val/test) ...")
    train_ds, val_ds, test_ds = get_datasets(cfg.dataset_root)

    # ---------------- Model A: Logistic Regression baseline ----------------
    print("\n=== Model A: Logistic Regression (逻辑回归) baseline ===")
    X_train, y_train = dataset_to_numpy_flat(train_ds)
    X_val, y_val = dataset_to_numpy_flat(val_ds)
    X_test, y_test = dataset_to_numpy_flat(test_ds)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)

    test_pred_a = clf.predict(X_test)
    metrics_a = compute_metrics(y_test, test_pred_a)
    print("Model A test metrics:", metrics_a)

    # ---------------- Model B: Small CNN baseline ----------------
    print("\n=== Model B: Small CNN (卷积神经网络) baseline ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model_b = train_cnn(train_loader, val_loader, device, epochs=cfg.cnn_epochs, lr=cfg.cnn_lr)
    metrics_b = eval_cnn(model_b, test_loader, device)
    print("Model B test metrics:", metrics_b)

    # Save results for report
    results = {
        "dataset": "BreastMNIST",
        "model_a": {"name": "LogisticRegression", "test": metrics_a},
        "model_b": {"name": "SmallCNN", "test": metrics_b},
        "notes": {
            "augmentation": "none (baseline)",
            "features_model_a": "flatten(28x28)=784 + StandardScaler",
            "training_budget_model_b": f"epochs={cfg.cnn_epochs}, batch_size={cfg.batch_size}, lr={cfg.cnn_lr}",
        },
    }
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {cfg.out_json}")


if __name__ == "__main__":
    main()
