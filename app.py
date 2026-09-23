from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "fraud_detection_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

data = pd.read_csv(DATA_PATH)
data = data.drop_duplicates()
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details below to check whether the transaction is potentially fraudulent.")
st.subheader("Transaction Details")

time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

st.write("Enter the V1 to V28 feature values:")

features = []

for i in range(1, 29):
    value = st.number_input(
        f"V{i}",
        value=0.0,
        format="%.6f"
    )
    features.append(value)
if st.button("🔍 Check Transaction"):

    input_data = [[
        time,
        *features,
        amount
    ]]

    input_df = pd.DataFrame(
        input_data,
        columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error("⚠️ Potential Fraudulent Transaction")
    else:
        st.success("✅ Transaction Classified as Normal")

    st.write(f"Fraud Probability: **{probability:.2%}**")
st.divider()

st.subheader("🧪 Test with a Real Dataset Transaction")

sample_type = st.selectbox(
    "Choose transaction type",
    ["Normal Transaction", "Fraudulent Transaction"]
)

if st.button("📊 Load & Test Real Transaction"):

    from sklearn.model_selection import train_test_split

    X_data = data.drop("Class", axis=1)
    y_data = data["Class"]

    _, X_test_sample, _, y_test_sample = train_test_split(
        X_data,
        y_data,
        test_size=0.2,
        random_state=42,
        stratify=y_data
    )

    desired_class = 0 if sample_type == "Normal Transaction" else 1

    available_indices = y_test_sample[
        y_test_sample == desired_class
    ].index

    sample_index = available_indices[0]

    sample = X_test_sample.loc[[sample_index]]
    actual_class = y_test_sample.loc[sample_index]

    sample_scaled = scaler.transform(sample)

    sample_prediction = model.predict(sample_scaled)[0]
    sample_probability = model.predict_proba(sample_scaled)[0][1]

    st.write("### Transaction Details")

    st.write(f"**Time:** {sample['Time'].iloc[0]:.2f}")
    st.write(f"**Amount:** ${sample['Amount'].iloc[0]:.2f}")

    st.write(f"**Actual Class:** {actual_class}")
    st.write(f"**Predicted Class:** {sample_prediction}")
    st.write(f"**Fraud Probability:** {sample_probability:.2%}")

    if sample_prediction == actual_class:
        st.success("✅ Model prediction matches the actual class.")
    else:
        st.warning("⚠️ Model prediction differs from the actual class.")