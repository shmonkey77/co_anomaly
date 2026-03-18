"""
Synthetic construction change order data generator.

Generates realistic line items across trades with intentionally injected
anomalies (inflated markups, wrong labor rates, unrealistic hours) so the
model has labeled examples to learn from.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Trade definitions ──────────────────────────────────────────────────────────
# Each trade has expected ranges for: labor_rate ($/hr), hours per unit,
# material_cost_pct (material as % of total), markup_pct
TRADES = {
    "electrical":   {"labor_rate": (65, 95),  "hours": (4, 40),   "material_pct": (0.30, 0.55), "markup": (0.08, 0.15)},
    "mechanical":   {"labor_rate": (70, 100), "hours": (6, 60),   "material_pct": (0.35, 0.60), "markup": (0.08, 0.15)},
    "concrete":     {"labor_rate": (55, 80),  "hours": (8, 80),   "material_pct": (0.40, 0.65), "markup": (0.05, 0.12)},
    "steel":        {"labor_rate": (75, 105), "hours": (10, 100), "material_pct": (0.50, 0.75), "markup": (0.05, 0.12)},
    "plumbing":     {"labor_rate": (70, 95),  "hours": (4, 50),   "material_pct": (0.30, 0.55), "markup": (0.08, 0.15)},
    "drywall":      {"labor_rate": (45, 70),  "hours": (6, 60),   "material_pct": (0.25, 0.45), "markup": (0.08, 0.15)},
    "hvac":         {"labor_rate": (75, 105), "hours": (8, 80),   "material_pct": (0.45, 0.70), "markup": (0.08, 0.15)},
    "general":      {"labor_rate": (50, 75),  "hours": (4, 40),   "material_pct": (0.20, 0.40), "markup": (0.10, 0.18)},
}

SCOPE_CATEGORIES = ["owner_directive", "unforeseen_condition", "design_error", "code_compliance", "value_engineering"]

TRADE_INDEX = {trade: i for i, trade in enumerate(TRADES)}
SCOPE_INDEX = {scope: i for i, scope in enumerate(SCOPE_CATEGORIES)}


def generate_normal_record(trade: str) -> dict:
    """Generate a realistic, non-anomalous line item for a given trade."""
    t = TRADES[trade]
    labor_rate    = np.random.uniform(*t["labor_rate"])
    hours         = np.random.uniform(*t["hours"])
    material_pct  = np.random.uniform(*t["material_pct"])
    markup_pct    = np.random.uniform(*t["markup"])

    labor_cost    = labor_rate * hours
    # derive material cost from the labor/material ratio
    material_cost = labor_cost * (material_pct / (1 - material_pct))
    subtotal      = labor_cost + material_cost
    total_cost    = subtotal * (1 + markup_pct)

    # small noise on labor burden (typically 28–35%)
    labor_burden_pct = np.random.uniform(0.28, 0.35)

    return {
        "trade":            trade,
        "scope_category":   np.random.choice(SCOPE_CATEGORIES),
        "labor_hours":      round(hours, 1),
        "labor_rate":       round(labor_rate, 2),
        "labor_cost":       round(labor_cost, 2),
        "labor_burden_pct": round(labor_burden_pct, 3),
        "material_cost":    round(material_cost, 2),
        "markup_pct":       round(markup_pct, 4),
        "total_cost":       round(total_cost, 2),
        "is_anomaly":       0,
        "anomaly_type":     "none",
    }


def inject_anomaly(record: dict) -> dict:
    """Take a normal record and corrupt one dimension to make it anomalous."""
    record = record.copy()
    anomaly_type = np.random.choice([
        "inflated_markup",
        "labor_rate_spike",
        "unrealistic_hours",
        "material_cost_spike",
        "labor_burden_inflation",
    ])

    if anomaly_type == "inflated_markup":
        # markup way above normal (30–80% instead of 8–18%)
        record["markup_pct"] = round(np.random.uniform(0.30, 0.80), 4)
        subtotal = record["labor_cost"] + record["material_cost"]
        record["total_cost"] = round(subtotal * (1 + record["markup_pct"]), 2)

    elif anomaly_type == "labor_rate_spike":
        # labor rate 2–4x the normal range for this trade
        t = TRADES[record["trade"]]
        record["labor_rate"] = round(np.random.uniform(t["labor_rate"][1] * 2, t["labor_rate"][1] * 4), 2)
        record["labor_cost"] = round(record["labor_rate"] * record["labor_hours"], 2)
        subtotal = record["labor_cost"] + record["material_cost"]
        record["total_cost"] = round(subtotal * (1 + record["markup_pct"]), 2)

    elif anomaly_type == "unrealistic_hours":
        # hours 5–10x the normal upper bound
        t = TRADES[record["trade"]]
        record["labor_hours"] = round(np.random.uniform(t["hours"][1] * 5, t["hours"][1] * 10), 1)
        record["labor_cost"]  = round(record["labor_rate"] * record["labor_hours"], 2)
        subtotal = record["labor_cost"] + record["material_cost"]
        record["total_cost"] = round(subtotal * (1 + record["markup_pct"]), 2)

    elif anomaly_type == "material_cost_spike":
        # material cost 3–6x what it should be for this labor amount
        record["material_cost"] = round(record["labor_cost"] * np.random.uniform(3, 6), 2)
        subtotal = record["labor_cost"] + record["material_cost"]
        record["total_cost"] = round(subtotal * (1 + record["markup_pct"]), 2)

    elif anomaly_type == "labor_burden_inflation":
        # labor burden far above normal (60–100% instead of 28–35%)
        record["labor_burden_pct"] = round(np.random.uniform(0.60, 1.00), 3)

    record["is_anomaly"]   = 1
    record["anomaly_type"] = anomaly_type
    return record


def generate_dataset(n_samples: int = 5000, anomaly_ratio: float = 0.20) -> pd.DataFrame:
    """
    Generate a labeled dataset.
    anomaly_ratio: fraction of records that are anomalous (default 20%)
    """
    records = []
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal    = n_samples - n_anomalies

    trades = list(TRADES.keys())

    for _ in range(n_normal):
        trade = np.random.choice(trades)
        records.append(generate_normal_record(trade))

    for _ in range(n_anomalies):
        trade = np.random.choice(trades)
        normal = generate_normal_record(trade)
        records.append(inject_anomaly(normal))

    df = pd.DataFrame(records)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset(n_samples=5000)

    out_path = os.path.join(os.path.dirname(__file__), "sample_output.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} records to {out_path}")
    print(f"\nClass balance:")
    print(df["is_anomaly"].value_counts())
    print(f"\nAnomaly type breakdown:")
    print(df[df["is_anomaly"] == 1]["anomaly_type"].value_counts())
    print(f"\nSample record (normal):")
    print(df[df["is_anomaly"] == 0].iloc[0].to_dict())
    print(f"\nSample record (anomaly):")
    print(df[df["is_anomaly"] == 1].iloc[0].to_dict())
