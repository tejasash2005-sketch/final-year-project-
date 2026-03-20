import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import bcrypt
import hashlib
import time
from sklearn.ensemble import IsolationForest
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Fintech Loan System", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.main {background: linear-gradient(135deg,#0f172a,#1e293b); color:white;}
.card {padding:15px;border-radius:12px;background:rgba(255,255,255,0.06);margin-bottom:10px;}
h1,h2,h3,h4 {color:white;}
</style>
""", unsafe_allow_html=True)

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

USERS_CSV = os.path.join(DATA_FOLDER, "users.csv")
APPLICANT_CSV = os.path.join(DATA_FOLDER, "loan.csv")

# ---------------- USERS ----------------
if not os.path.exists(USERS_CSV):
    pd.DataFrame([
        {"username":"admin","password_hash":bcrypt.hashpw("admin123".encode(),bcrypt.gensalt()).decode(),"role":"admin"},
        {"username":"user","password_hash":bcrypt.hashpw("user123".encode(),bcrypt.gensalt()).decode(),"role":"user"}
    ]).to_csv(USERS_CSV,index=False)

df_users = pd.read_csv(USERS_CSV)

# ---------------- FEATURES ----------------
FEATURE_LABELS = [
'Applicant Income','Coapplicant Income','Loan Amount','Credit History','Total Income',
'Loan-to-Income Ratio','Log Applicant Income','Log Coapplicant Income','Log Total Income',
'Loan per Coapplicant','DTI Ratio','Credit-Income Interaction','Applicant Income Squared',
'Loan Amount Squared','Income Ratio','Loan-Credit Interaction','High Loan Flag',
'High Income Flag','Coapplicant Flag','Loan Income Log Ratio','Sqrt Applicant Income',
'Sqrt Coapplicant Income','Applicant-Loan Interaction','Coapplicant-Loan Interaction',
'Marital Status Flag','Gender Flag','Age','Nationality Flag','Employment Status Flag'
]

# ---------------- LOANS ----------------
LOAN_PACKAGES = {
"Personal Loan": (0.12, 24),
"Home Loan": (0.08, 240),
"Car Loan": (0.10, 60),
"Education Loan": (0.09, 120),
"Gold Loan": (0.11, 12),
"Business Loan": (0.13, 60),
"Startup Loan": (0.15, 72),
"Travel Loan": (0.14, 12),
"Medical Loan": (0.10, 36),
"Agriculture Loan": (0.07, 60),
"10Y Plan": (0.09, 120),
"15Y Plan": (0.085, 180),
"20Y Plan": (0.08, 240),
"25Y Plan": (0.078, 300),
"30Y Plan": (0.075, 360)
}

# ---------------- MODEL ----------------
try:
    rf_model = pickle.load(open(os.path.join(MODEL_FOLDER,"rf_model.pkl"),"rb"))
    scaler = pickle.load(open(os.path.join(MODEL_FOLDER,"scaler.pkl"),"rb"))
except:
    rf_model=None
    scaler=None

# ---------------- INIT DATA ----------------
ALL_COLUMNS = FEATURE_LABELS + [
"Name","Age","Gender","Nationality","Marital Status",
"Aadhaar","PAN","CreditFile","BankFile",
"Prediction","Risk","Fraud","EMI","Loan Type","Loan Status","Credit Score"
]

if not os.path.exists(APPLICANT_CSV):
    pd.DataFrame(columns=ALL_COLUMNS).to_csv(APPLICANT_CSV,index=False)

df_app = pd.read_csv(APPLICANT_CSV)

# Ensure columns exist
for col in ALL_COLUMNS:
    if col not in df_app.columns:
        df_app[col] = None

# ---------------- SESSION ----------------
if "login" not in st.session_state:
    st.session_state.login=False
    st.session_state.role=""
    st.session_state.user=""

# ---------------- AUTH ----------------
def verify(u,p):
    if u in df_users['username'].values:
        row = df_users[df_users['username']==u].iloc[0]
        if bcrypt.checkpw(p.encode(), row["password_hash"].encode()):
            return True, row["role"]
    return False,None

# ---------------- LOGIN ----------------
def register_user(username, password, role="user"):
    if username in df_users["username"].values:
        return False

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    new_user = pd.DataFrame([{
        "username": username,
        "password_hash": hashed,
        "role": role
    }])

    new_user.to_csv(USERS_CSV, mode="a", header=False, index=False)
    return True
if not st.session_state.login:

    tab1, tab2 = st.tabs(["🔐 Login", "🆕 Register"])

    # ---------------- LOGIN ----------------
    with tab1:
        st.subheader("Login")

        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            ok, role = verify(u, p)
            if ok:
                st.session_state.login = True
                st.session_state.role = role
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid login")

    # ---------------- REGISTER ----------------
    with tab2:
        st.subheader("Register")

        new_u = st.text_input("New Username", key="reg_user")
        new_p = st.text_input("New Password", type="password", key="reg_pass")

        if st.button("Create Account"):
            if new_u and new_p:
                if register_user(new_u, new_p):
                    st.success("Account created ✅ Now login")
                else:
                    st.error("Username already exists")
            else:
                st.warning("Enter username & password")

# ---------------- FILE SAVE ----------------
def save_file(file):
    if file is None:
        return ""
    name = file.name
    h = hashlib.md5((name+str(time.time())).encode()).hexdigest()
    path = os.path.join(UPLOAD_FOLDER, h+"_"+name)
    with open(path,"wb") as f:
        f.write(file.getbuffer())
    return h+"_"+name
# ---------------- MAIN ----------------
if st.session_state.login:

    if st.session_state.role == "user":

        # -------- HEADER --------
        st.markdown(
            """
            <style>
            .header-container {
                text-align: top;
                padding: 25px;
                border-radius: 15px;
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: white;
                box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
                animation: fadeIn 1.2s ease-in-out;
            }

            .title {
                font-size: 36px;
                font-weight: 700;
                animation: slideDown 1s ease-in-out;
            }

            .subtitle {
                font-size: 18px;
                opacity: 0.8;
                animation: fadeIn 2s ease-in-out;
            }

            @keyframes fadeIn {
                from {opacity: 0;}
                to {opacity: 1;}
            }

            @keyframes slideDown {
                from {
                    transform: translateY(-30px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
            </style>

            <div class="header-container">
                <div class="title">🏦 Bank Loan Approval Prediction System</div>
                <div class="subtitle">AI-powered Loan Analysis & Risk Prediction</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"### 👋 Welcome {st.session_state.user}")

        # -------- SIDEBAR --------
        st.sidebar.title("📊 Loan Features")
        features = {}

        for f in FEATURE_LABELS:
            features[f] = st.sidebar.number_input(f, value=1000.0)

        # -------- PERSONAL INFO --------
        st.subheader("👤 Personal Info")
        c1, c2, c3, c4 = st.columns(4)

        name = c1.text_input("Name", key="name")
        age = c2.number_input("Age", 18, 100, 25, key="age")
        gender = c3.selectbox("Gender", ["Male", "Female"], key="gender")
        nationality = c4.text_input("Nationality", key="nat")

        marital = st.selectbox("Marital Status", ["Single", "Married"], key="marital")

        # -------- LOAN --------
        loan_type = st.selectbox("Loan Type", list(LOAN_PACKAGES.keys()))
        credit_score = st.slider("Credit Score", 300, 900, 700)

        # -------- DOCUMENTS --------
        st.subheader("📄 Documents")

        aadhaar = st.file_uploader("Aadhaar", key="aadhaar")
        pan = st.file_uploader("PAN", key="pan")
        credit = st.file_uploader("Credit Report", key="credit")
        bank = st.file_uploader("Bank Statement", key="bank")

        # -------- APPLY LOAN --------
        if st.button("Apply Loan"):

            X = np.array([list(features.values())])

            if rf_model:
                Xs = scaler.transform(X)
                pred = int(rf_model.predict_proba(Xs)[0][1] > 0.5)
            else:
                pred = 1

            rate, tenure = LOAN_PACKAGES[loan_type]
            r = rate / 12
            loan_amt = features["Loan Amount"]

            EMI = round(loan_amt * r * (1 + r) ** tenure / ((1 + r) ** tenure - 1), 2)

            risk = "Low" if credit_score > 750 else "Medium" if credit_score > 650 else "High"

            fraud = IsolationForest().fit(np.random.rand(50, 4)).predict([X[0][:4]])[0] == -1

            new_row = {col: features[col] for col in FEATURE_LABELS}

            new_row.update({
                "Name": name,
                "Age": age,
                "Gender": gender,
                "Nationality": nationality,
                "Marital Status": marital,
                "Aadhaar": save_file(aadhaar),
                "PAN": save_file(pan),
                "CreditFile": save_file(credit),
                "BankFile": save_file(bank),
                "Prediction": pred,
                "Risk": risk,
                "Fraud": fraud,
                "EMI": EMI,
                "Loan Type": loan_type,
                "Loan Status": "Under Review",
                "Credit Score": credit_score
            })

            df_app = pd.concat([df_app, pd.DataFrame([new_row])], ignore_index=True)
            df_app.to_csv(APPLICANT_CSV, index=False)

            st.success("Loan Applied ✅")

            # -------- EMI SCHEDULE --------
            st.subheader("📅 EMI Plan")

            bal = loan_amt
            schedule = []

            for m in range(1, tenure + 1):
                interest = bal * r
                principal = EMI - interest
                bal -= principal
                schedule.append([m, EMI, principal, interest, bal])

            st.dataframe(pd.DataFrame(schedule, columns=["Month", "EMI", "Principal", "Interest", "Balance"]))

         # ================= CHARTS (INSIDE USER BLOCK) =================
        st.subheader("📊 Loan Analytics & Predictions")

        c1, c2, c3 = st.columns(3)

        c1.plotly_chart(px.pie(df_app, names="Loan Type",
                               title="📌 Loan Type Prediction Distribution",
                               color_discrete_sequence=px.colors.sequential.Rainbow),
                        use_container_width=True)

        c2.plotly_chart(px.histogram(df_app, x="Risk",
                                     color="Risk",
                                     title="⚠️ Risk Prediction Distribution",
                                     color_discrete_sequence=px.colors.qualitative.Bold),
                        use_container_width=True)

        c3.plotly_chart(px.scatter(df_app, x="Total Income", y="Loan Amount",
                                   color="Credit Score",
                                   title="💰 Loan Approval Prediction vs Income",
                                   color_continuous_scale="Turbo"),
                        use_container_width=True)

        c4, c5, c6 = st.columns(3)

        c4.plotly_chart(px.bar(df_app, x="Loan Type",
                               color="Loan Type",
                               title="📊 Loan Type Prediction Count",
                               color_discrete_sequence=px.colors.qualitative.Vivid),
                        use_container_width=True)

        c5.plotly_chart(px.box(df_app, y="Credit Score",
                               color="Loan Type",
                               title="📈 Credit Score Prediction Spread"),
                        use_container_width=True)

        numeric_corr = df_app.select_dtypes(include=np.number).corr()

        c6.plotly_chart(px.imshow(numeric_corr,
                                  title="🔗 Feature Correlation (Model Insight)",
                                  color_continuous_scale="Plasma"),
                        use_container_width=True)


    else:

        st.title("🛠 Admin Dashboard")

        st.dataframe(df_app)

        st.subheader("📄 User Documents")
        st.write(df_app[["Name","Aadhaar","PAN","CreditFile","BankFile","Loan Status"]])

        for i in df_app.index:
            if st.button(f"Approve {i}"):
                df_app.at[i,"Loan Status"]="Approved"

        df_app.to_csv(APPLICANT_CSV,index=False)
        
# ==============================
# 💎 REAL FINTECH CARD (UPDATED)
# ==============================

if st.session_state.login and st.session_state.role == "user":

    user_df = df_app[df_app["Name"] == st.session_state.user]

    if not user_df.empty:

        latest = user_df.iloc[-1]

        with st.container():

            st.markdown("""
            <style>
            .stContainer {
                background: linear-gradient(135deg,#0f172a,#1e293b);
                padding: 20px;
                border-radius: 20px;
                box-shadow: 0px 15px 40px rgba(0,0,0,0.6);
                margin-top: 20px;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("### 💳 Loan Dashboard")

            # -------- TOP METRICS --------
            c1, c2, c3 = st.columns(3)

            c1.metric("💰 Loan Amount", f"₹ {latest.get('Loan Amount','N/A')}")
            c2.metric("📆 EMI", f"₹ {latest.get('EMI','N/A')}")
            c3.metric("📊 Credit Score", f"{latest.get('Credit Score','N/A')}")

            # -------- LOAN INFO --------
            st.markdown("#### 📄 Loan Info")
            col1, col2, col3 = st.columns(3)

            col1.info(f"Loan Type: {latest.get('Loan Type','N/A')}")
            col2.success(f"Status: {latest.get('Loan Status','N/A')}")
            col3.warning(f"Risk: {latest.get('Risk','N/A')}")

            # -------- LINE CHARTS (USING ALL FEATURES) --------
            st.markdown("#### 📊 Loan Feature Trends")

            # Prepare data
            feature_cols = [col for col in df_app.columns if col not in ["Name","Loan Status"]]

            numeric_df = df_app[feature_cols].select_dtypes(include=np.number)

            # Chart 1: Multi-feature trend
            st.plotly_chart(
                px.line(
                    numeric_df,
                    title="All Loan Features Trend (Multi-Line)",
                    labels={"value":"Value","index":"Record"}
                ),
                use_container_width=True
            )

            # Chart 2: Latest vs Average comparison
            avg_values = numeric_df.mean()

            compare_df = pd.DataFrame({
                "Feature": avg_values.index,
                "Average": avg_values.values,
                "Latest": numeric_df.iloc[-1].values if not numeric_df.empty else []
            })

            compare_df = compare_df.melt(id_vars="Feature", var_name="Type", value_name="Value")

            st.plotly_chart(
                px.line(
                    compare_df,
                    x="Feature",
                    y="Value",
                    color="Type",
                    markers=True,
                    title="Latest vs Average Feature Comparison"
                ),
                use_container_width=True
            )

    else:
        st.info("Apply loan to see dashboard")
