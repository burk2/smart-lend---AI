import streamlit as st
import pandas as pd
import joblib

# === PAGE SETTINGS ===
st.set_page_config(page_title="SmartLend AI", page_icon="💼")

# === INLINE STYLING ===
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #FF6600;
        font-size: 36px;
    }
    .stButton > button {
        background-color: #00BFFF;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.6em 2em;
    }
    </style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("<h1>💼 SmartLend – Loan Default Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("📈 Upload borrower data (CSV) and get instant predictions of default risk.")

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load("smartlend_model.pkl")

model = load_model()

# === FILE UPLOADER ===
uploaded_file = st.file_uploader("📁 Upload borrower data CSV", type="csv")

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        
        st.subheader("🔍 Data Preview")
        st.dataframe(data.head())

        # Predict
        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1]

        data["Default_Prediction"] = prediction
        data["Default_Probability"] = probability

        st.subheader("📊 Prediction Results")
        st.dataframe(data[["Default_Prediction", "Default_Probability"]])

        # Download
        csv_output = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv_output, file_name="smartlend_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("ℹ️ Please upload a CSV file to continue.")

# === FOOTER ===
st.markdown("---")
st.caption("🔐 Built by Nollin Masai Wabuti • SmartLend AI • 2025")
