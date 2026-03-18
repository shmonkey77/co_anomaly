# Construction Cost Anomaly Detector

A PyTorch neural network that flags suspicious line items in construction change orders — detecting outliers in labor rates, markup percentages, and unit costs compared to expected ranges for a given trade and scope.

Built as a companion to construction AI tooling, this model processes structured change order data and outputs an anomaly score (0–1) per line item.

## Project Structure

```
construction-anomaly-detector/
├── data/
│   ├── generate.py          # Synthetic training data generator
│   └── sample_output.csv    # Example generated dataset
├── model/
│   ├── dataset.py           # PyTorch Dataset class
│   ├── network.py           # Neural network architecture
│   └── train.py             # Training loop
├── api/
│   └── main.py              # FastAPI inference endpoint
├── tests/
│   └── test_model.py        # Unit tests
├── notebooks/
│   └── explore.ipynb        # EDA and training visualization
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Generate synthetic training data
python data/generate.py

# Train the model
python model/train.py

# Start the API
uvicorn api.main:app --reload
```

## Features

- Feedforward neural network (3 layers) trained on change order line item features
- Input features: labor hours, labor rate, material cost, markup %, trade type, scope category, total cost
- Output: anomaly score 0–1 (>0.5 flagged as anomalous)
- FastAPI inference endpoint
- Synthetic data generator with realistic construction cost distributions

## Tech Stack

Python, PyTorch, FastAPI, pandas, scikit-learn, pytest
