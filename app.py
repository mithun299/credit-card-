from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳",
                   layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Educational ML demo: estimate whether a transaction resembles the fraud class.")

model_path = Path(__file__).parent / "outputs" / "fraud_model.joblib"
if not model_path.exists():
    st.warning("Model not found. First run `python train.py` in the project folder.")
    st.stop()

bundle = joblib.load(model_path)
model = bundle["model"]
features = bundle["features"]

if not bundle.get("is_real_dataset", False):
    st.info("The current model was trained on synthetic demonstration data. "
            "Do not interpret its predictions as real banking risk assessments.")

st.subheader("Enter transaction features")
values = {}
cols = st.columns(2)
for i, feature in enumerate(features):
    with cols[i % 2]:
        values[feature] = st.number_input(
            feature, value=0.0, format="%.4f", key=f"feature_{i}"
        )

if st.button("Check Transaction", type="primary", use_container_width=True):
    row = pd.DataFrame([values], columns=features)
    prediction = int(model.predict(row)[0])
    probability = (float(model.predict_proba(row)[0, 1])
                   if hasattr(model, "predict_proba") else None)
    if prediction == 1:
        st.error("Model classification: FRAUD")
    else:
        st.success("Model classification: LEGITIMATE")
    if probability is not None:
        st.metric("Model-estimated fraud score", f"{probability:.2%}")
    st.caption("This is an educational model output, not a verified financial decision.")
