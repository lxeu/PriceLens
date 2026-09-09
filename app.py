import numpy as np
import streamlit as st

PARAMS_PATH = "models/params.npz"

@st.cache_data
def load_params(path=PARAMS_PATH):
    '''Load the trained model parameters.'''
    params = np.load(path)
    return params["w"], float(params["b"][0]), params["mu"], params["sigma"]

st.set_page_config(page_title="PriceLens", page_icon="🏠")

st.title("🏠 PriceLens")
st.caption("Edmonton housing price predictor — powered by gradient descent")