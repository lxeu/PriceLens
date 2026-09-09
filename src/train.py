import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import *
import os

FEATURES = ["Square Footage", "Bedrooms", "Bathrooms", "Year Built"]
TARGET = "Price"
CUTOFF = 1500000

# Data cleaning
data = pd.read_csv("data/edm_housing_data.csv")
for col in ["Price", "Square Footage"]:
    data[col] = pd.to_numeric(
        data[col].astype(str).str.replace(r"[\$,]", "", regex=True),
        errors="coerce"
    )

data = data[FEATURES + [TARGET]].dropna()

# Outlier cuts
data = data[
    (data["Price"] < CUTOFF) &
    (data["Square Footage"].between(300, 6000)) &
    (data["Year Built"] > 1900) &
    (data["Bedrooms"].between(1, 8))
]

x_train = data[FEATURES].to_numpy(dtype=float)
y_train = data[TARGET].to_numpy(dtype=float)

x_norm, mu, sigma = zscore_normalize_features(x_train)

w_init = np.zeros(x_norm.shape[1])
b_init = 0.0
alpha = 0.01
num_iters = 500

w, b, cost_history = gradient_descent(x_norm, y_train, w_init, b_init, alpha, num_iters)

print("b =", b)
print("weight values:")
for i in range(len(FEATURES)):
    print(FEATURES[i], "=", w[i])
print("final cost =", cost_history[-1])

# Save learned parameters
params_path = os.path.join("models", "params.npz")
os.makedirs("models", exist_ok=True)
np.savez(params_path, w=w, b=np.array([b]), mu=mu, sigma=sigma)
print(f"Parameters saved to {params_path}")

# Plot cost
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs Iteration")
plt.show()

# Plot predicted vs actual price
predictions = np.dot(x_norm, w) + b

plt.scatter(y_train, predictions, s=8, c="blue", alpha=0.6)
plt.plot([0, CUTOFF], [0, CUTOFF], color="red")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual")
plt.show()