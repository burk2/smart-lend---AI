import streamlit as st
import pandas as pd
import joblib

# === PAGE CONFIG ===
st.set_page_config(page_title="SmartLend AI", page_icon="💼", layout="centered")

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load("smartlend_model.pkl")

model = load_model()

# === HEADER ===
st.markdown("""
<div style="background-color:#e6f0f8;padding:1.5rem;border-radius:10px;">
    <h2 style="color:#005b96;">💼 SmartLend AI – Public Loan Risk Checker</h2>
    <p>Use the form below to check if a borrower is likely to default on a loan. Built for everyone, no files needed.</p>
</div>
""", unsafe_allow_html=True)

# === SIDEBAR BRANDING ===
with st.sidebar:
    st.image("https://img.icons8.com/color/96/loan.png", width=80)
    st.title("SmartLend AI")
    st.markdown("Built by **Nollin Masai Wabuti**")
    st.markdown("📧 masainollin@gmail.com")
    st.markdown("🌍 Kenya, 2025")

# === USER FORM ===
st.header("📝 Enter Borrower Details")

with st.form("predict_form"):
    age = st.number_input("Age", 18, 100, step=1)
    income = st.number_input("Monthly Income", min_value=0.0)
    debt_ratio = st.slider("Debt Ratio", 0.0, 5.0, step=0.01)
    num_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    revolving_util = st.slider("Revolving Utilization of Unsecured Lines", 0.0, 2.0, step=0.01)
    num_30_59 = st.number_input("30-59 Days Late (past)", 0, 10, step=1)
    num_60_89 = st.number_input("60-89 Days Late (past)", 0, 10, step=1)
    num_90 = st.number_input("90+ Days Late (past)", 0, 10, step=1)
    num_loans = st.number_input("Open Credit Lines and Loans", 0, 20, step=1)
    num_real_estate = st.number_input("Real Estate Loans", 0, 10, step=1)

    submitted = st.form_submit_button("💡 Predict Loan Risk")

if submitted:
    input_df = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": num_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": num_loans,
        "NumberOfTimes90DaysLate": num_90,
        "NumberRealEstateLoansOrLines": num_real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": num_60_89,
        "NumberOfDependents": num_dependents
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### ✅ Prediction Result")
    st.success(f"Prediction: **{'Will Default' if prediction == 1 else 'No Default'}**")
    st.metric("📊 Risk Probability", f"{probability:.2%}")

# === FOOTER ===
st.markdown("---")
st.caption("🔐 SmartLend AI • Powered by Nollin Masai • Streamlit • 2025")
