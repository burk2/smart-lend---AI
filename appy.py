import streamlit as st
import pandas as pd
import joblib
import os

#  Load custom CSS from external file 
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
#  Load the trained model 
MODEL_PATH = "smartlend_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

#  Streamlit App UI 
st.set_page_config(page_title="SmartLend AI", page_icon="💼")
st.title("💼 SmartLend – Loan Default Risk Predictor")
st.markdown("Upload borrower data and get instant default predictions with confidence scores.")

# File upload 
uploaded_file = st.file_uploader("📁 Upload borrower data CSV", type="csv")

if uploaded_file:
    try:
        # Read CSV
        data = pd.read_csv(uploaded_file)

        # Display preview
        st.subheader("📊 Borrower Data Preview")
        st.write(data.head())

        # Make predictions
        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1]

        # Append predictions to data
        data["Default_Prediction"] = prediction
        data["Default_Probability"] = probability

        # Show results
        st.subheader("✅ Prediction Results")
        st.dataframe(data[["Default_Prediction", "Default_Probability"]])

        # Option to download results
        csv_output = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv_output, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("Please upload a CSV file with borrower data.")

#  Footer 
st.markdown("---")
st.caption("🔐 Built by Nollin Masai Wabuti • SmartLend AI • 2025")
