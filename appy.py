import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load custom CSS safely
def local_css(file_name):
    with open(file_name, "r") as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

local_css("style.css")

# === Model Path ===
MODEL_PATH = "smartlend_model.pkl"

# === Load Model ===
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# === Page Setup ===
st.set_page_config(page_title="SmartLend AI", page_icon="💼")
st.title("💼 SmartLend – Loan Default Risk Predictor")
st.markdown("Upload borrower data and get instant default predictions **with explanations.**")

# === File Upload ===
uploaded_file = st.file_uploader("📁 Upload Borrower CSV", type="csv")

if uploaded_file:
    try:
        # Load and preview data
        data = pd.read_csv(uploaded_file)
        st.subheader("📊 Borrower Data Preview")
        st.dataframe(data.head())

        # Predict
        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1]

        # Format output
        data["Default_Prediction"] = np.where(prediction == 1, "❌ Default", "✅ No Default")
        data["Default_Probability (%)"] = (probability * 100).round(2)

        # Show prediction results
        st.subheader("✅ Prediction Results")
        st.dataframe(data)

        # Download option
        csv_output = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Results", csv_output, file_name="smartlend_predictions.csv", mime="text/csv")

        # === SHAP EXPLAINABILITY ===
        st.subheader("🧠 SHAP Explanation for One Prediction")

        row_index = st.number_input("🔍 Select a row to explain", min_value=0, max_value=len(data)-1, value=0)

        # Drop prediction columns before explanation
        features = data.drop(columns=["Default_Prediction", "Default_Probability (%)"])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        # Display SHAP prediction explanation
        st.markdown(f"**Prediction:** {data.iloc[row_index]['Default_Prediction']}  |  **Probability:** {data.iloc[row_index]['Default_Probability (%)']}%")

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.initjs()
        shap.force_plot(
            base_value=explainer.expected_value[1],
            shap_values=shap_values[1][row_index],
            features=features.iloc[row_index],
            matplotlib=True,
            show=False
        )
        st.pyplot(bbox_inches="tight")

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")

else:
    st.info("📌 Please upload a CSV file with borrower information to begin.")

# === Footer ===
st.markdown("---")
st.caption("🔐 Built by Nollin Masai Wabuti • SmartLend AI • 2025")
