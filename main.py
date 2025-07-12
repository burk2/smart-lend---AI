import streamlit as st
import pandas as pd
import joblib

# === PAGE CONFIG ===
st.set_page_config(page_title="SmartLend AI", page_icon="💼")

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load("smartlend_model.pkl")

model = load_model()

# === SIDEBAR ===
with st.sidebar:
    st.image("https://img.icons8.com/color/96/loan.png", width=80)
    st.markdown("## 💼 SmartLend AI")
    st.markdown("Built by **Nollin Masai Wabuti**")
    st.markdown("[🌐 GitHub](https://github.com/your-username)")
    st.markdown("📅 Version: 2025.1.0")
    st.markdown("🔒 Powered by ML & AI")

# === TABS ===
tab1, tab2, tab3 = st.tabs(["📁 Upload & Predict", "📊 Model Info", "ℹ️ About App"])

# ========== TAB 1: Upload & Predict ==========
with tab1:
    st.header("📁 Upload Borrower Data & Predict Risk")

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.subheader("🔍 Preview of Uploaded Data")
            st.dataframe(df.head())

            # Predict
            prediction = model.predict(df)
            probability = model.predict_proba(df)[:, 1]

            df["Default_Prediction"] = prediction
            df["Default_Probability"] = probability

            st.subheader("📊 Prediction Results")
            st.dataframe(df[["Default_Prediction", "Default_Probability"]])

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions", csv, file_name="smartlend_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("ℹ️ Upload a CSV file with borrower information to get started.")

# ========== TAB 2: Model Info ==========
with tab2:
    st.header("📊 About the Machine Learning Model")
    st.markdown("""
    - ✅ **Trained on:** Public credit risk dataset
    - 📈 **Model Type:** Random Forest Classifier
    - 🧠 **Features Used:** Income, Age, Past Due History, Loans
    - 🔍 **Label:** Loan Default within 2 years
    - 📊 **Output:** Prediction (`0` or `1`) + Probability score

    The model helps financial institutions quickly assess borrower risk.
    """)

# ========== TAB 3: About the App ==========
with tab3:
    st.header("ℹ️ About SmartLend AI")
    st.markdown("""
    **SmartLend AI** is a modern loan default prediction tool built for microfinance and fintech teams.

    - 🧠 Built by: **Nollin Masai Wabuti**
    - 🌍 Origin: Kenya
    - 💡 Goal: Empower risk management using AI
    - 🔗 GitHub: [Your Profile](https://github.com/your-username)
    - ☁️ Hosted on: [Streamlit Cloud](https://streamlit.io)

    Want to use this in production? Contact the developer.
    """)

# === FOOTER ===
st.markdown("---")
st.caption("© 2025 SmartLend AI — Created with ❤️ by Nollin Masai Wabuti")
