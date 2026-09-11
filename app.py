import streamlit as st
from src.predict import load_params, predict_price

@st.cache_data
def get_params():
    '''Load the trained model parameters.'''
    return load_params()

st.set_page_config(page_title="PriceLens", page_icon="🏠")

# Streamlit has no parameter to hide number_input steppers, so hide them directly.
st.markdown("""
<style>
[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

st.title("🏠 PriceLens")
st.caption("Edmonton housing price predictor")

st.divider()

col1, col2 = st.columns(2)

with col1:
    # Range comes from the training dataset cutoffs
    sqft = st.number_input("Square footage", min_value=300, max_value=6000, step=50, 
                                                    value=None, placeholder="Enter a value")
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=None,
                                                        placeholder="Enter a value")

with col2:
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=9, value=None,
                                                        placeholder="Enter a value")
    year_built = st.number_input("Year built", min_value=1900, max_value=2026, value=None,
                                                        placeholder="Enter a value")

st.space("xxlarge")

@st.dialog("Predicted")
def show_popup(price):
    st.space("small")
    st.markdown(
        f"<div style='text-align:center'>The predicted house price is "
        f"<strong>${price:,.0f}</strong></div>",
        unsafe_allow_html=True)
    st.space("medium")

left, mid, right = st.columns([1, 2, 1])
with mid:
    if st.button("Predict house price", type="primary", width="stretch"):
        if None in (sqft, bedrooms, bathrooms, year_built):
            st.error("Please fill in all four fields.")
        else:
            price = predict_price(sqft, bedrooms, bathrooms, year_built,
                                  params=get_params())
            show_popup(price)