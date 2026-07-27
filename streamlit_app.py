# these are all the libraries my app needs
import os
import joblib          # to load my saved model
import streamlit as st # the web app framework
import numpy as np
import pandas as pd

# i import my own preprocessing so the app cleans the input the SAME way i trained
from features import preprocess

# sets the browser tab title, icon, and keeps the layout centered
st.set_page_config(page_title="Purchase-Intent Predictor",layout="centered")


# this loads my trained model from the .pkl file
# @st.cache_resource means it only loads once, not every time i click a button (faster)
@st.cache_resource
def load_model():
    # i build the path relative to this file so it still works after i deploy
    path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
    artifact = joblib.load(path)
    # i saved the model AND the training columns together, so i unpack both here
    return artifact["model"], list(artifact["columns"])

# try/except so if the model file is missing, the user sees a clear message
# instead of the app crashing with a scary error
try:
    model, train_columns = load_model()
except Exception as e:
    st.error("Could not load the model file. Make sure "
             "models/model.pkl is present. "
             f"Details: {e}")
    st.stop()   # stop the app here if there's no model to use


# the title and a short line explaining what the app does
st.title(" Purchase-Intent Predictor")
st.caption("Enter what a live website visitor is doing, and the model predicts whether "
           "the session will end in a purchase, so the team can nudge high-intent "
           "shoppers at the right moment.")


# i made 2 preset scenarios so i can fill all the inputs in one click during my demo.
# "High-intent shopper" should predict Buy, "Casual browser" should predict No buy.
PRESETS = {
    "Custom (enter your own below)": None,
    "High-intent shopper": dict(
        Administrative=3, Administrative_Duration=80.0, Informational=1,
        Informational_Duration=20.0, ProductRelated=40, ProductRelated_Duration=1200.0,
        BounceRates=0.005, ExitRates=0.01, PageValues=35.0, SpecialDay=0.0,
        Month="Nov", VisitorType="Returning_Visitor", Weekend="True",
        OperatingSystems=2, Browser=2, Region=1, TrafficType=2),
    "Casual browser": dict(
        Administrative=0, Administrative_Duration=0.0, Informational=0,
        Informational_Duration=0.0, ProductRelated=2, ProductRelated_Duration=15.0,
        BounceRates=0.2, ExitRates=0.2, PageValues=0.0, SpecialDay=0.0,
        Month="Feb", VisitorType="New_Visitor", Weekend="False",
        OperatingSystems=1, Browser=1, Region=3, TrafficType=1),
}
# the dropdown to pick a preset
preset_name = st.selectbox("Start from a scenario", list(PRESETS.keys()))
# P holds the chosen preset's values, which i use as the default for each input below.
# if the user picks "Custom" (None), i fall back to the high-intent numbers as defaults
P = PRESETS[preset_name] or PRESETS["High-intent shopper"]

# the month and visitor options the model was trained on
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITORS = ["Returning_Visitor", "New_Visitor", "Other"]


# i used plain english labels here (not the raw column names) so a marketing person
# can understand it, which is what the rubric wants for the target audience
st.subheader("Session behaviour")
# split the form into 2 columns so it's not one long list
c1, c2 = st.columns(2)
with c1:
    # number_input(label, min, max, default) makes a box that only accepts numbers in range
    product_related = st.number_input("Product pages viewed", 0, 800,
                                       int(P["ProductRelated"]))
    product_dur = st.number_input("Time on product pages (sec)", 0.0, 60000.0,
                                   float(P["ProductRelated_Duration"]))
    admin = st.number_input("Account/admin pages viewed", 0, 100,
                            int(P["Administrative"]))
    admin_dur = st.number_input("Time on admin pages (sec)", 0.0, 6000.0,
                                float(P["Administrative_Duration"]))
    info = st.number_input("Info pages viewed", 0, 100, int(P["Informational"]))
    info_dur = st.number_input("Time on info pages (sec)", 0.0, 6000.0,
                               float(P["Informational_Duration"]))
with c2:
    # help= adds a little (?) tooltip explaining the field
    page_values = st.number_input("Page Value (Google Analytics)", 0.0, 400.0,
                                  float(P["PageValues"]),
                                  help="Average value of the pages visited this session. "
                                       "The strongest single signal in the model.")
    # slider is nicer than a box for values between 0 and 1
    bounce = st.slider("Bounce rate", 0.0, 1.0, float(P["BounceRates"]), 0.005)
    exit_rate = st.slider("Exit rate", 0.0, 1.0, float(P["ExitRates"]), 0.005)
    special_day = st.slider("Closeness to a special day", 0.0, 1.0,
                            float(P["SpecialDay"]), 0.2,
                            help="1.0 = right on a holiday like Valentine's Day.")
    month = st.selectbox("Month", MONTHS, index=MONTHS.index(P["Month"]))
    visitor = st.selectbox("Visitor type", VISITORS, index=VISITORS.index(P["VisitorType"]))
    # radio = pick one option, horizontal so True/False sit side by side
    weekend = st.radio("Weekend session?", ["True", "False"],
                       index=0 if P["Weekend"] == "True" else 1, horizontal=True)

# i hide the confusing technical codes inside an expander so the main form stays clean
with st.expander("Advanced (technical session attributes)"):
    a1, a2, a3, a4 = st.columns(4)
    op_sys = a1.number_input("OS code", 1, 8, int(P["OperatingSystems"]))
    browser = a2.number_input("Browser code", 1, 13, int(P["Browser"]))
    region = a3.number_input("Region code", 1, 9, int(P["Region"]))
    traffic = a4.number_input("Traffic type", 1, 20, int(P["TrafficType"]))


# the widgets already stop out of range numbers, but they can't catch weird COMBINATIONS,
# so i add my own checks here and collect any problems in this list
errors = []
# time was spent but no pages viewed makes no sense
if (product_related + admin + info) == 0 and (product_dur + admin_dur + info_dur) > 0:
    errors.append("Time was recorded but no pages were viewed, please check the inputs.")
# bounce rate should never be higher than exit rate for a real session
if bounce > exit_rate + 1e-9:
    errors.append("Bounce rate cannot be higher than exit rate for a session.")


# this runs only when the user clicks the Predict button
if st.button("Predict purchase intent", type="primary"):
    # if my checks found problems, show them and don't predict
    if errors:
        for msg in errors:
            st.error(msg)
    else:
        # try/except so a bad input shows a message instead of crashing the app
        try:
            # put all the inputs into one dictionary, then into a 1-row dataframe
            row = dict(
                Administrative=admin, Administrative_Duration=admin_dur,
                Informational=info, Informational_Duration=info_dur,
                ProductRelated=product_related, ProductRelated_Duration=product_dur,
                BounceRates=bounce, ExitRates=exit_rate, PageValues=page_values,
                SpecialDay=special_day, OperatingSystems=op_sys, Browser=browser,
                Region=region, TrafficType=traffic, Month=month,
                VisitorType=visitor, Weekend=weekend)
            df_input = pd.DataFrame([row])

            # run the SAME preprocessing as training (engineer features + one-hot encode)
            df_input = preprocess(df_input)
            # one row only creates its own category columns, so i reindex to the full
            # training columns and fill the missing ones with 0, so it matches the model
            df_input = df_input.reindex(columns=train_columns, fill_value=0)

            # predict() gives 0 or 1 (buy or not), predict_proba() gives the probability
            will_buy = bool(model.predict(df_input)[0])
            proba = float(model.predict_proba(df_input)[0, 1])
        except Exception as e:
            st.error(f"Prediction failed, please check the inputs. Details: {e}")
            st.stop()

        # a progress bar as a visual of the probability
        st.progress(min(int(proba * 100), 100))
        # show a different message + recommended action depending on the prediction
        if will_buy:
            st.success(f" Likely to purchase, {proba*100:.0f}% probability")
            st.write("**Recommended action:** trigger a real-time nudge "
                     "(free-shipping banner, live-chat offer, or small discount).")
        else:
            st.info(f" Unlikely to purchase, {proba*100:.0f}% probability")
            st.write("**Recommended action:** hold incentives and let the visitor browse "
                     "to avoid wasting discounts.")

        # 3 metric tiles summarising the result
        m1, m2, m3 = st.columns(3)
        m1.metric("Purchase probability", f"{proba*100:.1f}%")
        m2.metric("Model decision", "Buy" if will_buy else "No buy")
        m3.metric("Suggested action", "Nudge" if will_buy else "Wait")

# a small footer line at the bottom of the page
st.caption("Model: tuned scikit-learn classifier chosen against a baseline. "
           "UCI Online Shoppers dataset. Same feature engineering as training.")
