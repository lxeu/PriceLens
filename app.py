import streamlit as st
from src.predict import load_params, predict_price

@st.cache_data
def get_params():
    '''Load the trained model parameters.'''
    return load_params()

st.set_page_config(page_title="PriceLens", page_icon="🏠")

st.html("""
<style>
[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"] {
    display: none;
}

[data-testid="stMetricValue"] {
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    font-weight: 300;
    letter-spacing: -0.03em;
}
</style>

<div style='text-align:center; margin-bottom:2.5rem'>
  <h1 style='font-family:Georgia, serif; font-size:4rem; font-weight:600;
             letter-spacing:-0.02em; margin:0 0 0.6rem 0'>PriceLens</h1>
  <p style='color:#8b949e; margin:0'>Estimate the market value of an Edmonton home</p>
</div>
""")

price = None
st.space("small")

with st.container(border=True):

    st.subheader(":material/tune: Property details")
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


    if st.button("Predict house price", type="primary", width="stretch"):
        if None in (sqft, bedrooms, bathrooms, year_built):
            st.error("Please fill in all four fields.")
        else:
            price = predict_price(sqft, bedrooms, bathrooms, year_built,
                                params=get_params())
            
if price is not None:
    st.space("medium")
    with st.container(border=True):
        st.metric("Estimated Value", f"{price:,.0f}")

st.space("medium")
with st.expander("How this works", icon=":material/info:"):
    st.markdown("""PriceLens estimates home values using linear regression trained on real Edmonton sales data.

The model looks at four things: square footage, bedrooms, bathrooms, and year built. Each one gets a weight during training, reflecting how much it pushes the price up or down. An estimate is the combination of those four weighted values.

Training runs on gradient descent. The model starts with rough guesses, measures how far off its predictions are against actual sale prices, adjusts the weights to reduce that error, and repeats until the predictions are accurate.""")

st.caption(
    "Built with NumPy · "
    "[Source](https://github.com/lxeu/PriceLens)"
)