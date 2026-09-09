import numpy as np

PARAMS_PATH = "models/params.npz"


def load_params(path=PARAMS_PATH):
    """Load the trained model parameters."""
    params = np.load(path)
    return params["w"], float(params["b"][0]), params["mu"], params["sigma"]


def predict_price(sqft, bedrooms, bathrooms, year_built, params=None):
    """
    Predict a single home's price.
    Takes raw (un-normalized) feature values and returns a dollar amount.
    """
    if params is None:
        params = load_params()
    w, b, mu, sigma = params

    x = np.array([sqft, bedrooms, bathrooms, year_built], dtype=float)
    x_norm = (x - mu) / sigma
    return float(np.dot(x_norm, w) + b)