import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from model import zscore_normalize_features, compute_cost, gradient_descent

FEATURES = ["Square Footage", "Bedrooms", "Bathrooms", "Year Built"]
TARGET = "Price"
CUTOFF = 1500000
ALPHA = 0.1   # 0.01 (what train.py uses) is too slow: weights are still 10-30% off after 500 iterations
NUM_ITERS = 500
SEED = 42  # fixed so the train/test split (and your numbers) are the same every run


def load_data():
    """Same cleaning as train.py."""
    data = pd.read_csv("data/edm_housing_data.csv")
    for col in ["Price", "Square Footage"]:
        data[col] = pd.to_numeric(
            data[col].astype(str).str.replace(r"[\$,]", "", regex=True),
            errors="coerce"
        )
    data = data[FEATURES + [TARGET]].dropna()
    data = data[
        (data["Price"] < CUTOFF) &
        (data["Square Footage"].between(300, 6000)) &
        (data["Year Built"] > 1900) &
        (data["Bedrooms"].between(1, 8))
    ]
    return data[FEATURES].to_numpy(dtype=float), data[TARGET].to_numpy(dtype=float)


def run_gd(x, y, alpha, num_iters):
    """Your gradient_descent, with its progress printing hidden."""
    with contextlib.redirect_stdout(io.StringIO()):
        return gradient_descent(x, y, np.zeros(x.shape[1]), 0.0, alpha, num_iters)


def iters_to_converge(cost_history, best_cost, tolerance=0.01):
    """First iteration where cost is within 1% of the best possible cost, or None."""
    for i, cost in enumerate(cost_history):
        if cost <= best_cost * (1 + tolerance):
            return i + 1
    return None


x, y = load_data()
print(f"Listings after cleaning: {len(y)}")

# 80/20 train/test split
rng = np.random.default_rng(SEED)
order = rng.permutation(len(y))
split = int(0.8 * len(y))
train_idx, test_idx = order[:split], order[split:]
x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

# Normalize with TRAINING stats only, then apply the same mu/sigma to the test set
x_train_norm, mu, sigma = zscore_normalize_features(x_train)
x_test_norm = (x_test - mu) / sigma

# Match against scikit-learn
w, b, cost_history = run_gd(x_train_norm, y_train, ALPHA, NUM_ITERS)

sk = LinearRegression().fit(x_train_norm, y_train)
best_cost = compute_cost(x_train_norm, y_train, sk.coef_, sk.intercept_)

my_pred = x_test_norm @ w + b
sk_pred = sk.predict(x_test_norm)
pred_gap = np.mean(np.abs(my_pred - sk_pred) / np.abs(sk_pred)) * 100

print("\n1. Match against scikit-learn")
print(f"{'Feature':<16}{'Mine':>14}{'scikit-learn':>16}")
for name, mine, theirs in zip(FEATURES, w, sk.coef_):
    print(f"{name:<16}{mine:>14,.0f}{theirs:>16,.0f}")
print(f"{'Bias':<16}{b:>14,.0f}{sk.intercept_:>16,.0f}")
weight_gap = np.max(np.abs(w - sk.coef_) / np.abs(sk.coef_)) * 100
print(f"Largest weight difference: {weight_gap:.2f}%")
print(f"Average difference in test predictions: {pred_gap:.2f}%")

normalized_iters = iters_to_converge(cost_history, best_cost)

RAW_ALPHA = 3e-7
RAW_ITERS = 2000
_, _, raw_history = run_gd(x_train, y_train, RAW_ALPHA, RAW_ITERS)
raw_best = compute_cost(x_train_norm, y_train, sk.coef_, sk.intercept_)
raw_iters = iters_to_converge(raw_history, raw_best)

print("\n2. Iterations to get within 1% of the best possible cost")
print(f"Normalized (alpha={ALPHA}): {normalized_iters}")
if raw_iters is None:
    gap = (raw_history[-1] / raw_best - 1) * 100
    print(f"Unnormalized (alpha={RAW_ALPHA}): did not converge in {RAW_ITERS} iterations "
          f"(still {gap:.0f}% above best cost)")
else:
    print(f"Unnormalized (alpha={RAW_ALPHA}): {raw_iters}")

mae = np.mean(np.abs(my_pred - y_test))
baseline_mae = np.mean(np.abs(np.mean(y_train) - y_test))

print("\n3. Error on held-out listings (never seen in training)")
print(f"Test listings: {len(y_test)}")
print(f"Mean absolute error: ${mae:,.0f}")
print(f"Baseline (always guess the average price): ${baseline_mae:,.0f}")
print(f"Improvement over baseline: {(1 - mae / baseline_mae) * 100:.0f}%")