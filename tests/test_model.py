"""
Unit tests for data generation, dataset encoding, and model forward pass.
Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.generate import (
    generate_normal_record, inject_anomaly, generate_dataset,
    TRADES, TRADE_INDEX, SCOPE_INDEX
)
from model.dataset import ChangeOrderDataset, INPUT_DIM
from model.network import AnomalyDetector


# ── Data generation tests ──────────────────────────────────────────────────────

class TestDataGeneration:

    def test_normal_record_fields(self):
        record = generate_normal_record("electrical")
        required = ["trade", "scope_category", "labor_hours", "labor_rate",
                    "labor_cost", "material_cost", "markup_pct", "total_cost",
                    "labor_burden_pct", "is_anomaly", "anomaly_type"]
        for field in required:
            assert field in record, f"Missing field: {field}"

    def test_normal_record_not_anomaly(self):
        for trade in TRADES:
            record = generate_normal_record(trade)
            assert record["is_anomaly"] == 0
            assert record["anomaly_type"] == "none"

    def test_normal_record_costs_positive(self):
        record = generate_normal_record("mechanical")
        assert record["labor_hours"] > 0
        assert record["labor_rate"] > 0
        assert record["labor_cost"] > 0
        assert record["material_cost"] > 0
        assert record["total_cost"] > 0

    def test_anomaly_injection_flips_label(self):
        for trade in TRADES:
            normal  = generate_normal_record(trade)
            anomaly = inject_anomaly(normal)
            assert anomaly["is_anomaly"] == 1
            assert anomaly["anomaly_type"] != "none"

    def test_dataset_class_balance(self):
        df = generate_dataset(n_samples=1000, anomaly_ratio=0.20)
        assert len(df) == 1000
        anomaly_pct = df["is_anomaly"].mean()
        assert 0.17 <= anomaly_pct <= 0.23, f"Anomaly ratio out of range: {anomaly_pct:.2f}"

    def test_all_trades_present(self):
        df = generate_dataset(n_samples=2000)
        for trade in TRADES:
            assert trade in df["trade"].values, f"Trade missing: {trade}"


# ── Dataset encoding tests ─────────────────────────────────────────────────────

class TestDataset:

    @pytest.fixture
    def small_df(self):
        return generate_dataset(n_samples=100)

    def test_input_dim(self, small_df):
        ds = ChangeOrderDataset(small_df, fit_scaler=True)
        X, y = ds[0]
        assert X.shape == (INPUT_DIM,), f"Expected ({INPUT_DIM},), got {X.shape}"

    def test_labels_binary(self, small_df):
        ds = ChangeOrderDataset(small_df, fit_scaler=True)
        for i in range(len(ds)):
            _, y = ds[i]
            assert y.item() in {0.0, 1.0}

    def test_scaler_reuse(self, small_df):
        train_df = small_df.iloc[:70]
        val_df   = small_df.iloc[70:]
        train_ds = ChangeOrderDataset(train_df, fit_scaler=True)
        # Should not raise — reuses trained scaler
        val_ds = ChangeOrderDataset(val_df, scaler=train_ds.scaler)
        assert len(val_ds) == 30

    def test_no_nan_in_features(self, small_df):
        ds = ChangeOrderDataset(small_df, fit_scaler=True)
        for i in range(len(ds)):
            X, _ = ds[i]
            assert not torch.isnan(X).any(), f"NaN in features at index {i}"


# ── Model tests ────────────────────────────────────────────────────────────────

class TestModel:

    def test_output_shape(self):
        model = AnomalyDetector()
        x = torch.randn(16, INPUT_DIM)
        out = model(x)
        assert out.shape == (16,), f"Expected (16,), got {out.shape}"

    def test_output_range(self):
        model = AnomalyDetector()
        x = torch.randn(100, INPUT_DIM)
        out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_single_sample(self):
        model = AnomalyDetector()
        model.eval()
        x = torch.randn(1, INPUT_DIM)
        out = model(x)
        assert out.shape == (1,)

    def test_gradients_flow(self):
        model = AnomalyDetector()
        x = torch.randn(8, INPUT_DIM)
        y = torch.randint(0, 2, (8,)).float()
        criterion = torch.nn.BCELoss()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
