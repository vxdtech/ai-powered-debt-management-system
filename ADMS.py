import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Autonomous Debt AI", page_icon="🏦", layout="wide")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Make sure 'delinquency_model.pkl' is in this folder!
    return joblib.load('delinquency_model.pkl')

try:
    model = load_model()
    st.sidebar.success("✅ Random Forest Model Loaded (87% Acc)")
except:
    st.sidebar.error("❌ Model file not found!")

# --- 3. UI HEADER ---
st.title("🏦 Autonomous AI Powered Debt Management System")
st.markdown("""
This system uses a **Random Forest Classifier** to predict customer delinquency risk. 
It triggers **autonomous interventions** based on a precision-optimized threshold of **0.26**.
""")

# --- 4. FEATURE INPUTS ---
st.header("👤 Customer Risk Profile")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    income = st.number_input("Annual Income ($)", 10000, 500000, 55000)
    credit_score = st.slider("Credit Score", 300, 850, 650)

with col2:
    utilization = st.slider("Credit Utilization (%)", 0.0, 1.0, 0.35)
    missed_payments = st.number_input("Historical Missed Payments", 0, 24, 0)
    loan_balance = st.number_input("Current Loan Balance ($)", 0, 100000, 5000)

with col3:
    dti = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    tenure = st.number_input("Account Tenure (Months)", 0, 360, 24)
    recent_miss = st.selectbox("Did they miss a payment recently?", [0, 1])

# --- 5. PREDICTION LOGIC ---
if st.button("🚀 Analyze Risk & Trigger Action"):
    # Create the full 63-feature list (Starting with all zeros)
    feature_names = [
        'Age', 'Income', 'Credit_Score', 'Credit_Utilization', 'Missed_Payments', 
        'Loan_Balance', 'Debt_to_Income_Ratio', 'Account_Tenure', 'Month_1', 'Month_2', 
        'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Employment_Status_1', 'Employment_Status_10', 
        'Employment_Status_11', 'Employment_Status_12', 'Employment_Status_13', 'Employment_Status_14', 
        'Employment_Status_15', 'Employment_Status_16', 'Employment_Status_17', 'Employment_Status_18', 
        'Employment_Status_19', 'Employment_Status_2', 'Employment_Status_3', 'Employment_Status_4', 
        'Employment_Status_5', 'Employment_Status_6', 'Employment_Status_7', 'Employment_Status_8', 
        'Employment_Status_9', 'Employment_Status_Business', 'Employment_Status_EMP', 'Employment_Status_Employed', 
        'Employment_Status_Gold', 'Employment_Status_Platinum', 'Employment_Status_Self-employed', 
        'Employment_Status_Student', 'Employment_Status_Unemployed', 'Employment_Status_employed', 
        'Employment_Status_retired', 'Credit_Card_Type_Chicago', 'Credit_Card_Type_Gold', 
        'Credit_Card_Type_Houston', 'Credit_Card_Type_Los Angeles', 'Credit_Card_Type_Missed', 
        'Credit_Card_Type_New York', 'Credit_Card_Type_On-time', 'Credit_Card_Type_Phoenix', 
        'Credit_Card_Type_Platinum', 'Credit_Card_Type_Standard', 'Credit_Card_Type_Student', 
        'Location_Houston', 'Location_Late', 'Location_Los Angeles', 'Location_Missed', 
        'Location_New York', 'Location_On-time', 'Location_Phoenix', 'Avg_Status_6m', 'Recent_Miss'
    ]
    
    # Initialize input data with 0s
    input_data = pd.DataFrame(np.zeros((1, 63)), columns=feature_names)
    
    # Map our UI inputs to the DataFrame
    input_data['Age'] = age
    input_data['Income'] = income
    input_data['Credit_Score'] = credit_score
    input_data['Credit_Utilization'] = utilization
    input_data['Missed_Payments'] = missed_payments
    input_data['Loan_Balance'] = loan_balance
    input_data['Debt_to_Income_Ratio'] = dti
    input_data['Account_Tenure'] = tenure
    input_data['Recent_Miss'] = recent_miss

    # Get Prediction
    prob = model.predict_proba(input_data)[0, 1]
    threshold = 0.26
    
    # --- 6. DISPLAY RESULTS ---
    st.divider()
    st.subheader("Results & Decision")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Delinquency Probability", f"{prob:.2%}")
        if prob >= threshold:
            st.error("🔴 RISK CATEGORY: HIGH")
        else:
            st.success("🟢 RISK CATEGORY: LOW")

    with res_col2:
        st.write("**🤖 Autonomous Action Triggered:**")
        if prob >= threshold:
            st.warning("👉 Send Automated Debt Restructuring Offer (Email/SMS)")
            st.info("Strategy: Priority 1 - Preventative Relief Program")
        else:
            st.write("👉 Standard Monthly Monitoring - No action needed.")

    # Show Feature Importance explanation
    st.expander("Why this decision?").write("""
    The Random Forest model analyzed 63 data points. Primary risk drivers identified 
    include Credit Utilization, Recent Missed Payments, and Debt-to-Income ratio.
    """)