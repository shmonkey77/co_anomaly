"""
Training loop for the construction anomaly detector.

Trains for N epochs, evaluates on validation set each epoch,
saves the best checkpoint, and prints a final test set report.

Usage:
    python model/train.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.dataset import load_splits
from model.network import AnomalyDetector

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_output.csv")
CHECKPOINT   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model.pt")
EPOCHS       = 40
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD    = 0.5   # anomaly score > THRESHOLD → flagged


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss  = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            probs = model(X)
            loss  = criterion(probs, y)
            total_loss += loss.item() * len(y)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc, np.array(all_probs), np.array(all_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_splits(DATA_PATH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AnomalyDetector().to(device)

    # Use pos_weight to handle class imbalance (80% normal, 20% anomaly)
    pos_weight = torch.tensor([4.0]).to(device)  # ~(1 - anomaly_ratio) / anomaly_ratio
    criterion  = nn.BCELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_auc = 0.0

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val AUC':>10}")
    print("-" * 44)

    for epoch in range(1, EPOCHS + 1):
        train_loss              = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {val_auc:>10.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     val_auc,
                "scaler":      train_ds.scaler,
            }, CHECKPOINT)
            print(f"  ✓ Saved checkpoint (val AUC: {val_auc:.4f})")

    # ── Test evaluation ────────────────────────────────────────────────────────
    print(f"\n{'─' * 44}")
    print("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, test_auc, probs, labels = evaluate(model, test_loader, criterion, device)
    preds = (probs >= THRESHOLD).astype(int)

    print(f"\nTest AUC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Normal", "Anomaly"]))


if __name__ == "__main__":
    main()
