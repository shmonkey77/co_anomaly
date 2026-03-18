"""
PyTorch Dataset for construction change order line items.

Handles feature encoding (label encoding for categoricals, normalization
for numerics) and exposes a standard Dataset interface for the DataLoader.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.generate import TRADE_INDEX, SCOPE_INDEX

# Numeric features the model will consume
NUMERIC_FEATURES = [
    "labor_hours",
    "labor_rate",
    "labor_cost",
    "labor_burden_pct",
    "material_cost",
    "markup_pct",
    "total_cost",
]

INPUT_DIM = len(NUMERIC_FEATURES) + len(TRADE_INDEX) + len(SCOPE_INDEX)


class ChangeOrderDataset(Dataset):
    """
    Encodes a DataFrame of change order line items into tensors.

    Categorical features (trade, scope_category) are one-hot encoded.
    Numeric features are standardized using the provided scaler
    (fit on training set, reused for val/test).
    """

    def __init__(self, df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = False):
        self.df = df.copy().reset_index(drop=True)

        # ── Encode categoricals ───────────────────────────────────────────────
        n = len(df)

        trade_enc = np.zeros((n, len(TRADE_INDEX)), dtype=np.float32)
        for i, trade in enumerate(df["trade"]):
            trade_enc[i, TRADE_INDEX[trade]] = 1.0

        scope_enc = np.zeros((n, len(SCOPE_INDEX)), dtype=np.float32)
        for i, scope in enumerate(df["scope_category"]):
            scope_enc[i, SCOPE_INDEX[scope]] = 1.0

        # ── Encode numerics ───────────────────────────────────────────────────
        numeric = df[NUMERIC_FEATURES].values.astype(np.float32)

        if scaler is None:
            scaler = StandardScaler()

        if fit_scaler:
            numeric = scaler.fit_transform(numeric)
        else:
            numeric = scaler.transform(numeric)

        self.scaler = scaler

        # ── Concatenate all features ──────────────────────────────────────────
        self.X = np.concatenate([numeric, trade_enc, scope_enc], axis=1)
        self.y = df["is_anomaly"].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def load_splits(csv_path: str, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Load CSV, split into train/val/test, fit scaler on train only.
    Returns (train_ds, val_ds, test_ds).
    """
    df = pd.read_csv(csv_path)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_df  = df.iloc[:n_test]
    val_df   = df.iloc[n_test:n_test + n_val]
    train_df = df.iloc[n_test + n_val:]

    train_ds = ChangeOrderDataset(train_df, fit_scaler=True)
    val_ds   = ChangeOrderDataset(val_df,   scaler=train_ds.scaler)
    test_ds  = ChangeOrderDataset(test_df,  scaler=train_ds.scaler)

    print(f"Split sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_ds, val_ds, test_ds
