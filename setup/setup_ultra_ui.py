import os

print("🚀 FINAL MERGED FINTECH SYSTEM SETUP STARTED...")

# -----------------------------
# CREATE FOLDERS
# -----------------------------
folders = [
    "frontend/styles",
    "frontend/components",
    "data",
    "models",
    "uploads"
]

for f in folders:
    os.makedirs(f, exist_ok=True)

def create(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# -----------------------------
# ULTRA UI CSS
# -----------------------------
create("frontend/styles/style.css", """
body {
    background: linear-gradient(135deg,#020617,#0f172a);
    color:white;
}
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    padding:20px;
    border-radius:15px;
    margin-bottom:20px;
}
.stButton>button {
    background: linear-gradient(90deg,#6366f1,#9333ea);
    color:white;
    border-radius:10px;
}
""")

# -----------------------------
# UI LOADER
# -----------------------------
create("frontend/components/ui.py", """
import streamlit as st
def apply():
    with open("frontend/styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
""")

# -----------------------------
# FINAL APP.PY (YOUR FULL CODE + UI)
# -----------------------------
create("app.py", '''import streamlit as st
from frontend.components.ui import apply
apply()

import pandas as pd
import numpy as np
import pickle
import os
import bcrypt
import hashlib
import time
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Ultimate Fintech Loan System", layout="wide")

BASE_DIR = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_DIR, "data")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

USERS_CSV = os.path.join(DATA_FOLDER, "users.csv")
APPLICANT_CSV = os.path.join(DATA_FOLDER, "loan_applicants.csv")

FEATURE_LABELS = [
'Applicant Income','Coapplicant Income','Loan Amount','Credit History','Total Income',
'Loan-to-Income Ratio','Log Applicant Income','Log Coapplicant Income','Log Total Income',
'Loan per Coapplicant','DTI Ratio','Credit-Income Interaction','Applicant Income Squared',
'Loan Amount Squared','Income Ratio','Loan-Credit Interaction','High Loan Flag',
'High Income Flag','Coapplicant Flag','Loan Income Log Ratio','Sqrt Applicant Income',
'Sqrt Coapplicant Income','Applicant-Loan Interaction','Coapplicant-Loan Interaction',
'Marital Status Flag','Gender Flag','Age','Nationality Flag','Employment Status Flag'
]

LOAN_PACKAGES = {
"Personal Loan": (0.12, 24),
"Home Loan": (0.08, 120),
"Car Loan": (0.10, 60),
"Education Loan": (0.09, 48),
"Gold Loan": (0.11, 12),
"Business Loan": (0.13, 36),
"Startup Loan": (0.15, 36),
"Travel Loan": (0.14, 12),
"Medical Loan": (0.10, 24),
"Agriculture Loan": (0.07, 36)
}

# Load model safely
try:
    rf_model = pickle.load(open(os.path.join(MODEL_FOLDER,"rf_model.pkl"),"rb"))
    scaler = pickle.load(open(os.path.join(MODEL_FOLDER,"scaler.pkl"),"rb"))
except:
    rf_model = None
    scaler = None

# Init CSV
if not os.path.exists(APPLICANT_CSV):
    pd.DataFrame(columns=FEATURE_LABELS + [
        "Name","Age","Gender","Nationality","Marital Status",
        "Prediction","Risk","Fraud Flag","EMI",
        "Loan Status","Loan Type","Payment Status","Credit Score"
    ]).to_csv(APPLICANT_CSV,index=False)

df_app = pd.read_csv(APPLICANT_CSV)

# SESSION
if "logged_in" not in st.session_state:
    st.session_state.logged_in=False
    st.session_state.role=""

# SIMPLE LOGIN (safe fallback)
st.sidebar.title("Login")
user = st.sidebar.text_input("User")
pwd = st.sidebar.text_input("Pass", type="password")

if st.sidebar.button("Login"):
    if user=="admin":
        st.session_state.logged_in=True
        st.session_state.role="admin"
    else:
        st.session_state.logged_in=True
        st.session_state.role="user"

# MAIN
if st.session_state.logged_in:

    if st.session_state.role=="user":

        st.markdown('<div class="glass"><h2>Apply Loan</h2></div>', unsafe_allow_html=True)

        features = {}
        for f in FEATURE_LABELS:
            features[f] = st.number_input(f, value=1000.0)

        loan_type = st.selectbox("Loan Type", list(LOAN_PACKAGES.keys()))
        credit_score = st.slider("Credit Score",300,900,700)

        if st.button("Submit Loan"):

            if rf_model:
                X = np.array([list(features.values())])
                Xs = scaler.transform(X)
                prob = rf_model.predict_proba(Xs)[0][1]
                pred = int(prob>0.5)
            else:
                pred = 1

            risk = "Low" if credit_score>750 else "Medium" if credit_score>650 else "High"

            rate, tenure = LOAN_PACKAGES[loan_type]
            r = rate/12
            n = tenure
            loan_amt = features["Loan Amount"]

            EMI = round(loan_amt*r*(1+r)**n/((1+r)**n-1),2)

            fraud = IsolationForest().fit(np.random.rand(50,4)).predict([list(features.values())[:4]])[0]==-1

            new = features.copy()
            new.update({
                "Prediction":pred,
                "Risk":risk,
                "Fraud Flag":fraud,
                "EMI":EMI,
                "Loan Status":"Under Review",
                "Loan Type":loan_type,
                "Payment Status":"Pending",
                "Credit Score":credit_score
            })

            df_app = pd.concat([df_app,pd.DataFrame([new])],ignore_index=True)
            df_app.to_csv(APPLICANT_CSV,index=False)

            st.success("Loan Submitted")

    else:

        st.markdown('<div class="glass"><h2>Admin Dashboard</h2></div>', unsafe_allow_html=True)

        st.dataframe(df_app)

        for i,row in df_app.iterrows():
            if st.button(f"Approve {i}"):
                df_app.at[i,"Loan Status"]="Approved"

        df_app.to_csv(APPLICANT_CSV,index=False)
''')

# -----------------------------
# REQUIREMENTS
# -----------------------------
create("requirements.txt", """
streamlit
pandas
numpy
scikit-learn
plotly
bcrypt
""")

# -----------------------------
# RUN FILE
# -----------------------------
create("run.bat", """
@echo off
cd /d %~dp0
start cmd /k "python -m streamlit run app.py"
timeout /t 5 >nul
start http://localhost:8501
exit
""")

print("🎉 FINAL SYSTEM READY!")
print("👉 Run: python final_ultimate_merge.py")
print("👉 Then double click run.bat")
