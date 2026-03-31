import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import bcrypt
import hashlib
import time
import random
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="bank loan approval prediction", layout="wide", page_icon="🤖")

# styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #03001e, #07002e, #0a0a2e, #0d1b3e);
    font-family: 'Orbitron', sans-serif;
    color: #00ffe0;
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(160deg, #0a0a1a, #0d0d2b, #0f0f3d);
    color: #00ffe0;
    border-right: 1px solid #00ffe044;
}
.stMetric > div { color: #00ffe0 !important; font-weight: bold; }
h1,h2,h3,h4,p { color: #00ffe0 !important; font-weight: bold; }
input, select, textarea {
    background-color: rgba(0,0,0,0.4) !important;
    color: #00ffe0 !important;
    border: 1px solid #00ffe066 !important;
    border-radius: 10px !important;
}
.emi-section {
    background: linear-gradient(135deg, #1a003a, #2d005f, #1a0040, #0d0028);
    border: 1px solid #bf00ff88;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 0 40px #bf00ff55, inset 0 0 60px #6600aa22;
    margin-bottom: 20px;
}
.analytics-section {
    background: linear-gradient(135deg, #001a3a, #002855, #001f45, #000d2e);
    border: 1px solid #0077ff88;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 0 40px #0055ff44, inset 0 0 60px rgba(0,34,153,0.07);
    margin-bottom: 20px;
}
.dashboard-section {
    background: linear-gradient(135deg, #001200, #003300, #001a00, #000f00);
    border: 1px solid #00ff6644;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 0 40px #00ff4422, inset 0 0 80px #00440033;
    margin-bottom: 20px;
}
.loan-card {
    background: linear-gradient(135deg, #002233, #003344, #001122);
    border: 1px solid #00ffe055;
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 0 50px #00ffe022, inset 0 0 60px #00aaaa11;
    margin-bottom: 20px;
}
.advanced-card {
    background: linear-gradient(135deg, #1a0a00, #2d1500, #1a0800);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 0 50px #ff660033, inset 0 0 60px #ff440011;
    border: 1px solid #ff660055;
    margin-bottom: 15px;
}
.lifecycle-section {
    background: linear-gradient(135deg, #001a1a, #002b2b, #001f1f, #000f0f);
    border: 1px solid #00ffe088;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 0 40px #00ffe033, inset 0 0 60px #00aaaa11;
    margin-bottom: 20px;
}
.explain-section {
    background: linear-gradient(135deg, #1a0020, #2d0040, #1a0030, #0d0020);
    border: 1px solid #ff00cc88;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 0 40px #ff00cc44, inset 0 0 60px #aa006622;
    margin-bottom: 20px;
}
.heatmap-section {
    background: linear-gradient(135deg, #1a0800, #2d1400, #1a0800, #0d0400);
    border: 1px solid #ff660088;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 0 40px #ff440033, inset 0 0 60px #ff220011;
    margin-bottom: 20px;
}
.kyc-section {
    background: linear-gradient(135deg, #001a00, #002b00, #001a0a, #000f05);
    border: 1px solid #00ff8888;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 0 40px #00ff4433, inset 0 0 60px #00aa4411;
    margin-bottom: 20px;
}
.payment-section {
    background: linear-gradient(135deg, #1a1a00, #2b2b00, #1a1500, #0f0d00);
    border: 1px solid #ffff0088;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 0 40px #ffff0033, inset 0 0 60px #aaaa0011;
    margin-bottom: 20px;
}
.gateway-section {
    background: linear-gradient(135deg, #001133, #002255, #001a44, #000d22);
    border: 1px solid #3399ff88;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 0 40px #3399ff33, inset 0 0 60px #002266aa;
    margin-bottom: 20px;
}
.receipt-card {
    background: linear-gradient(135deg, #002200, #003300, #001a00);
    border: 2px solid #00ff88;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 0 30px #00ff4444;
    margin-bottom: 15px;
}
.lc-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    margin: 5px 0;
    border-radius: 12px;
    border-left: 4px solid;
    font-size: 13px;
    letter-spacing: 1px;
}
.done-step   { background:#00ff8811; border-color:#00ff88; color:#00ff88; }
.active-step { background:#ffff0011; border-color:#ffff00; color:#ffff00; }
.wait-step   { background:#ffffff08; border-color:#444; color:#888; }
.overdue-emi { background:#ff000011; border:1px solid #ff0000aa; border-radius:10px; padding:8px 14px; color:#ff6666; }
.due-emi     { background:#ffaa0011; border:1px solid #ffaa00aa; border-radius:10px; padding:8px 14px; color:#ffcc44; }
.paid-emi    { background:#00ff8811; border:1px solid #00ff88aa; border-radius:10px; padding:8px 14px; color:#00ff88; }
.upcoming-emi{ background:#3399ff11; border:1px solid #3399ffaa; border-radius:10px; padding:8px 14px; color:#66bbff; }
</style>
""", unsafe_allow_html=True)

# folder setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPL_DIR  = os.path.join(BASE_DIR, "uploads")
MDL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPL_DIR, exist_ok=True)

USERS_FILE    = os.path.join(DATA_DIR, "users.csv")
LOANS_FILE    = os.path.join(DATA_DIR, "loan.csv")
PAYMENTS_FILE = os.path.join(DATA_DIR, "payments.csv")
KYC_FILE      = os.path.join(DATA_DIR, "kyc.csv")

# create default users if not exist
if not os.path.exists(USERS_FILE):
    default_users = [
        {"username": "admin", "password_hash": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(), "role": "admin"},
        {"username": "user",  "password_hash": bcrypt.hashpw("user123".encode(),  bcrypt.gensalt()).decode(), "role": "user"}
    ]
    pd.DataFrame(default_users).to_csv(USERS_FILE, index=False)

users_df = pd.read_csv(USERS_FILE)

# init payments file
PAYMENT_COLS = ["Username", "Loan_Index", "Month", "Due_Date", "Paid_Date",
                "Principal", "Interest", "EMI_Amount", "Late_Fee", "Total_Paid",
                "Payment_Method", "Transaction_ID", "Status"]
if not os.path.exists(PAYMENTS_FILE):
    pd.DataFrame(columns=PAYMENT_COLS).to_csv(PAYMENTS_FILE, index=False)

# init kyc file
KYC_COLS = ["Username", "Phone", "OTP_Verified", "Selfie_Uploaded",
            "Account_Number", "IFSC_Code", "Bank_Name", "KYC_Status",
            "Verified_At"]
if not os.path.exists(KYC_FILE):
    pd.DataFrame(columns=KYC_COLS).to_csv(KYC_FILE, index=False)

# all feature names the model uses
FEATURES = [
    'Applicant Income', 'Coapplicant Income', 'Loan Amount', 'Credit History',
    'Total Income', 'Loan-to-Income Ratio', 'Log Applicant Income',
    'Log Coapplicant Income', 'Log Total Income', 'Loan per Coapplicant',
    'DTI Ratio', 'Credit-Income Interaction', 'Applicant Income Squared',
    'Loan Amount Squared', 'Income Ratio', 'Loan-Credit Interaction',
    'High Loan Flag', 'High Income Flag', 'Coapplicant Flag',
    'Loan Income Log Ratio', 'Sqrt Applicant Income', 'Sqrt Coapplicant Income',
    'Applicant-Loan Interaction', 'Coapplicant-Loan Interaction',
    'Marital Status Flag', 'Gender Flag', 'Age', 'Nationality Flag',
    'Employment Status Flag'
]

# loan types with rate and tenure
LOANS = {
    "Personal Loan":    (0.12, 24),
    "Home Loan":        (0.08, 240),
    "Car Loan":         (0.10, 60),
    "Education Loan":   (0.09, 120),
    "Gold Loan":        (0.11, 12),
    "Business Loan":    (0.13, 60),
    "Startup Loan":     (0.15, 72),
    "Travel Loan":      (0.14, 12),
    "Medical Loan":     (0.10, 36),
    "Agriculture Loan": (0.07, 60),
    "10Y Plan":         (0.09, 120),
    "15Y Plan":         (0.085, 180),
    "20Y Plan":         (0.08, 240),
    "25Y Plan":         (0.078, 300),
    "30Y Plan":         (0.075, 360)
}

# try load model
try:
    rf_model = pickle.load(open(os.path.join(MDL_DIR, "rf_model.pkl"), "rb"))
    scaler   = pickle.load(open(os.path.join(MDL_DIR, "scaler.pkl"), "rb"))
except:
    rf_model = None
    scaler = None

# all columns for loans csv
ALL_COLS = FEATURES + [
    "Username", "Name", "Age", "Gender", "Nationality", "Marital Status",
    "Aadhaar", "PAN", "CreditFile", "BankFile",
    "Prediction", "Risk", "Fraud", "EMI", "Loan Type", "Loan Status",
    "Credit Score", "Applied Date", "Last Updated",
    "Lifecycle Stage", "Approval Probability", "Explainability",
    "EMI_Day", "Disbursement_Date", "Account_Number", "Bank_Name", "IFSC_Code"
]

if not os.path.exists(LOANS_FILE):
    pd.DataFrame(columns=ALL_COLS).to_csv(LOANS_FILE, index=False)

loans_df = pd.read_csv(LOANS_FILE)
for c in ALL_COLS:
    if c not in loans_df.columns:
        loans_df[c] = None

# session defaults
for k, v in [
    ("login", False), ("role", ""), ("username", ""),
    ("show_adv", False), ("otp_sent", False), ("otp_code", ""),
    ("otp_verified", False), ("kyc_step", 1),
    ("payment_gateway_open", False), ("last_receipt", None)
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== HELPER FUNCTIONS =====================

def check_login(u, p):
    if u in users_df["username"].values:
        row = users_df[users_df["username"] == u].iloc[0]
        if bcrypt.checkpw(p.encode(), row["password_hash"].encode()):
            return True, row["role"]
    return False, None

def add_user(u, p):
    if u in users_df["username"].values:
        return False
    h = bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
    pd.DataFrame([{"username": u, "password_hash": h, "role": "user"}]).to_csv(
        USERS_FILE, mode="a", header=False, index=False)
    return True

def store_upload(f):
    if f is None:
        return ""
    uid = hashlib.md5((f.name + str(time.time())).encode()).hexdigest()
    dest = os.path.join(UPL_DIR, uid + "_" + f.name)
    with open(dest, "wb") as out:
        out.write(f.getbuffer())
    return uid + "_" + f.name

def emi_calc(principal, annual_rate, months):
    r = annual_rate / 12
    if r == 0:
        return round(principal / months, 2)
    return round(principal * r * (1+r)**months / ((1+r)**months - 1), 2)

def chart_style(title, bg="#000a1e", tc="#5599ff"):
    return dict(
        title=dict(text=title, font=dict(color=tc, size=13, family="Orbitron")),
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=tc, family="Orbitron"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                   zeroline=False, tickfont=dict(color=tc)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                   zeroline=False, tickfont=dict(color=tc)),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(0,255,68,0.2)",
                    borderwidth=1, font=dict(color=tc)),
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified"
    )

LC_STEPS = [
    ("Application Submitted", "📋"),
    ("Document Verification", "🔍"),
    ("AI Risk Assessment",    "🤖"),
    ("Credit Bureau Check",   "🏦"),
    ("Underwriting Review",   "📝"),
    ("Final Approval",        "✅"),
    ("Loan Disbursement",     "💰"),
    ("Active Repayment",      "🔄"),
    ("Loan Closed",           "🎉"),
]

def get_stage_idx(status):
    mp = {
        "Under Review": 2, "Approved": 5,
        "Disbursed": 6, "Active": 7, "Closed": 8, "Rejected": 3
    }
    return mp.get(status, 0)

def render_lifecycle(cur_idx):
    html = "<div style='padding:6px 0;'>"
    for i, (lbl, ico) in enumerate(LC_STEPS):
        if i < cur_idx:
            cls, badge = "done-step", "✔"
        elif i == cur_idx:
            cls, badge = "active-step", "▶"
        else:
            cls, badge = "wait-step", str(i+1)
        html += f"<div class='lc-step {cls}'><span style='font-size:17px'>{ico}</span><span style='min-width:26px;font-weight:bold'>[{badge}]</span><span>{lbl}</span></div>"
    html += "</div>"
    return html

def get_explain_scores(feat_vals, cscore, lamount):
    scores = {
        "Credit Score":         cscore / 900,
        "Applicant Income":     min(feat_vals.get("Applicant Income", 0) / 200000, 1),
        "DTI Ratio":            1 - min(feat_vals.get("DTI Ratio", 0.5), 1),
        "Loan Amount":          1 - min(lamount / 5000000, 1),
        "Credit History":       feat_vals.get("Credit History", 0),
        "Coapplicant Income":   min(feat_vals.get("Coapplicant Income", 0) / 100000, 1),
        "Total Income":         min(feat_vals.get("Total Income", 0) / 300000, 1),
        "Loan-to-Income Ratio": 1 - min(feat_vals.get("Loan-to-Income Ratio", 2) / 10, 1),
    }
    return scores

def generate_otp():
    return str(random.randint(100000, 999999))

def get_emi_due_dates(disbursement_date_str, emi_day, num_months):
    """Generate list of due dates for each EMI month."""
    try:
        base = datetime.strptime(disbursement_date_str[:10], "%Y-%m-%d")
    except:
        base = datetime.now()
    dates = []
    for m in range(1, num_months + 1):
        due = base + relativedelta(months=m)
        due = due.replace(day=min(emi_day, 28))
        dates.append(due)
    return dates

def get_emi_status(due_date):
    today = datetime.now().date()
    due   = due_date.date() if hasattr(due_date, 'date') else due_date
    if today > due:
        return "Overdue"
    elif today == due:
        return "Due Today"
    elif (due - today).days <= 7:
        return "Due Soon"
    else:
        return "Upcoming"

def calc_late_fee(due_date, rate_per_day=50):
    today = datetime.now().date()
    due   = due_date.date() if hasattr(due_date, 'date') else due_date
    days_late = (today - due).days
    return max(0, days_late * rate_per_day)

def load_payments():
    df = pd.read_csv(PAYMENTS_FILE)
    for c in PAYMENT_COLS:
        if c not in df.columns:
            df[c] = None
    return df

def load_kyc():
    df = pd.read_csv(KYC_FILE)
    for c in KYC_COLS:
        if c not in df.columns:
            df[c] = None
    return df

def get_user_kyc(username):
    df = load_kyc()
    rows = df[df["Username"] == username]
    if rows.empty:
        return None
    return rows.iloc[-1]

def save_kyc(record: dict):
    df = load_kyc()
    df = df[df["Username"] != record["Username"]]
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(KYC_FILE, index=False)

def cyber_theme(fig):
    fig.update_layout(
        paper_bgcolor="#050d1a",
        plot_bgcolor="#050d1a",
        font=dict(color="#00ffe0", size=12),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,255,224,0.2)",
                   zeroline=False, color="#00ffe0"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,255,224,0.2)",
                   zeroline=False, color="#00ffe0"),
        legend=dict(font=dict(color="#00ffe0"))
    )
    return fig

# ===================== LOGIN PAGE =====================
if not st.session_state.login:
    t1, t2 = st.tabs(["🔐 Login", "🆕 Register"])
    with t1:
        inp_u = st.text_input("Username")
        inp_p = st.text_input("Password", type="password")
        if st.button("Login"):
            ok, role = check_login(inp_u, inp_p)
            if ok:
                st.session_state.login    = True
                st.session_state.role     = role
                st.session_state.username = inp_u.strip().lower()
                st.rerun()
            else:
                st.error("Wrong username or password")
    with t2:
        nu  = st.text_input("Pick a Username")
        np_ = st.text_input("Pick a Password", type="password")
        if st.button("Create Account"):
            if add_user(nu, np_):
                st.success("Account created! Go login.")
            else:
                st.error("Username already taken")
    st.stop()

# ===================== USER SECTION =====================
if st.session_state.login and st.session_state.role == "user":
    menu = st.sidebar.radio("📂 Navigation", [
        "🏠 Loan Application",
        "🔐 KYC Verification",
        "💳 EMI Payment Center",
        "📄 Loan Details (Real Bank View)"
    ])

    # =====================================================================
    # PAGE: KYC VERIFICATION
    # =====================================================================
    if menu == "🔐 KYC Verification":
        st.title("🔐 KYC — Know Your Customer")
        st.markdown("Complete your KYC to enable loan disbursement to your bank account.")

        kyc_rec = get_user_kyc(st.session_state.username)
        kyc_verified = kyc_rec is not None and str(kyc_rec.get("KYC_Status", "")) == "Verified"

        if kyc_verified:
            st.success(f"✅ KYC Verified on {kyc_rec.get('Verified_At', 'N/A')}")
            st.markdown(f"""
            <div class='kyc-section'>
            <h3 style='color:#00ff88!important;'>✅ KYC Summary</h3>
            <p>📱 Phone: {kyc_rec.get('Phone','N/A')} &nbsp;|&nbsp; ✔ OTP: {'Verified' if kyc_rec.get('OTP_Verified') else 'No'}</p>
            <p>🏦 Bank: {kyc_rec.get('Bank_Name','N/A')} &nbsp;|&nbsp; IFSC: {kyc_rec.get('IFSC_Code','N/A')}</p>
            <p>💳 Account: {'*'*8 + str(kyc_rec.get('Account_Number',''))[-4:] if kyc_rec.get('Account_Number') else 'N/A'}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔄 Re-do KYC"):
                kyc_verified = False
                st.session_state.otp_verified = False
                st.session_state.otp_sent     = False
                st.session_state.otp_code     = ""
                st.rerun()
        else:
            st.markdown('<div class="kyc-section">', unsafe_allow_html=True)

            # ---- STEP 1: Phone + OTP ----
            st.markdown("### 📱 Step 1 — Phone Verification")
            phone_in = st.text_input("Enter Mobile Number", max_chars=10, placeholder="10-digit number")

            col_s1, col_s2 = st.columns(2)
            if col_s1.button("📨 Send OTP"):
                if len(phone_in) == 10 and phone_in.isdigit():
                    otp = generate_otp()
                    st.session_state.otp_code = otp
                    st.session_state.otp_sent = True
                    st.success(f"✅ OTP sent to +91-{phone_in}  (Demo OTP: **{otp}**)")
                else:
                    st.error("Enter a valid 10-digit number.")

            if st.session_state.otp_sent and not st.session_state.otp_verified:
                otp_in = st.text_input("Enter OTP", max_chars=6, placeholder="6-digit OTP")
                if col_s2.button("✔ Verify OTP"):
                    if otp_in == st.session_state.otp_code:
                        st.session_state.otp_verified = True
                        st.success("✅ OTP Verified Successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect OTP. Try again.")

            if st.session_state.otp_verified:
                st.success("✅ Phone Verified")

            st.markdown("---")

            # ---- STEP 2: Selfie Upload ----
            st.markdown("### 🤳 Step 2 — Face Verification (Selfie)")
            selfie_f = st.file_uploader("Upload a clear selfie / photo for face verification",
                                        type=["jpg","jpeg","png"], key="selfie_up")
            selfie_ok = selfie_f is not None
            if selfie_ok:
                st.image(selfie_f, caption="Uploaded Photo", width=200)
                st.success("✅ Selfie uploaded. (Face match: Demo mode — auto-passed)")

            st.markdown("---")

            # ---- STEP 3: Bank Account ----
            st.markdown("### 🏦 Step 3 — Bank Account Details")
            st.info("Loan amount will be credited to this account after approval.")
            ba_col1, ba_col2 = st.columns(2)
            acc_num   = ba_col1.text_input("Account Number", max_chars=18)
            ifsc_code = ba_col2.text_input("IFSC Code", max_chars=11, placeholder="e.g. SBIN0001234")
            bank_name = st.selectbox("Bank Name", [
                "State Bank of India", "HDFC Bank", "ICICI Bank",
                "Axis Bank", "Punjab National Bank", "Bank of Baroda",
                "Canara Bank", "Kotak Mahindra Bank", "Yes Bank",
                "IndusInd Bank", "Union Bank", "Other"
            ])

            st.markdown("---")

            # ---- SUBMIT KYC ----
            if st.button("🚀 Submit KYC", key="submit_kyc"):
                errs = []
                if not st.session_state.otp_verified:
                    errs.append("Phone OTP not verified.")
                if not selfie_ok:
                    errs.append("Selfie not uploaded.")
                if not acc_num or len(acc_num) < 9:
                    errs.append("Enter valid account number.")
                if not ifsc_code or len(ifsc_code) < 11:
                    errs.append("Enter valid IFSC code (11 chars).")

                if errs:
                    for e in errs:
                        st.error(f"❌ {e}")
                else:
                    rec = {
                        "Username":       st.session_state.username,
                        "Phone":          phone_in,
                        "OTP_Verified":   True,
                        "Selfie_Uploaded": store_upload(selfie_f),
                        "Account_Number": acc_num,
                        "IFSC_Code":      ifsc_code.upper(),
                        "Bank_Name":      bank_name,
                        "KYC_Status":     "Verified",
                        "Verified_At":    datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    save_kyc(rec)
                    st.success("🎉 KYC Verified Successfully!")
                    st.balloons()
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================================
    # PAGE: EMI PAYMENT CENTER
    # =====================================================================
    elif menu == "💳 EMI Payment Center":
        st.title("💳 EMI Payment Center")

        loans_df = pd.read_csv(LOANS_FILE)
        for c in ALL_COLS:
            if c not in loans_df.columns:
                loans_df[c] = None

        user_loans = loans_df[loans_df["Username"] == st.session_state.username]

        if user_loans.empty:
            st.warning("No loan applications found. Apply for a loan first.")
            st.stop()

        # pick loan
        loan_options = [f"Loan #{i} — {row['Loan Type']} — ₹{row['Loan Amount']:,} — {row['Loan Status']}"
                        for i, row in user_loans.iterrows()]
        sel_loan_lbl = st.selectbox("Select Loan", loan_options)
        sel_idx = int(sel_loan_lbl.split("#")[1].split(" ")[0])
        sel_loan = loans_df.iloc[sel_idx]

        ltype   = str(sel_loan.get("Loan Type", "Personal Loan"))
        rate, t = LOANS.get(ltype, (0.12, 24))
        l_amt   = float(sel_loan.get("Loan Amount", 0) or 0)
        emi_amt = float(sel_loan.get("EMI", emi_calc(l_amt, rate, int(t))) or 0)
        emi_day = int(sel_loan.get("EMI_Day", 5) or 5)
        disb_dt = str(sel_loan.get("Disbursement_Date", sel_loan.get("Applied Date", datetime.now().strftime("%Y-%m-%d"))) or datetime.now().strftime("%Y-%m-%d"))

        # load payments for this loan
        pay_df = load_payments()
        my_pays = pay_df[(pay_df["Username"] == st.session_state.username) &
                         (pay_df["Loan_Index"] == sel_idx)]

        paid_months = set(my_pays["Month"].astype(int).tolist())

        # generate schedule
        due_dates = get_emi_due_dates(disb_dt, emi_day, int(t))

        bal_run = l_amt
        r_mo    = rate / 12
        sched_rows = []
        for mo in range(1, int(t)+1):
            int_p = bal_run * r_mo
            pri_p = emi_amt - int_p
            bal_run = max(bal_run - pri_p, 0)
            due = due_dates[mo-1]
            status = "Paid" if mo in paid_months else get_emi_status(due)
            late_fee = 0 if mo in paid_months else calc_late_fee(due)
            sched_rows.append({
                "Month":         mo,
                "Due Date":      due.strftime("%Y-%m-%d"),
                "EMI (₹)":       round(emi_amt, 2),
                "Principal (₹)": round(pri_p, 2),
                "Interest (₹)":  round(int_p, 2),
                "Balance (₹)":   round(bal_run, 2),
                "Late Fee (₹)":  late_fee,
                "Status":        status,
            })

        sched_df = pd.DataFrame(sched_rows)

        # summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 Total Loan",   f"₹ {l_amt:,.0f}")
        m2.metric("📆 Monthly EMI",  f"₹ {emi_amt:,.2f}")
        m3.metric("✅ EMIs Paid",     f"{len(paid_months)} / {int(t)}")
        overdue_c = len(sched_df[(sched_df["Status"] == "Overdue") & (~sched_df["Month"].isin(paid_months))])
        m4.metric("⚠️ Overdue EMIs", str(overdue_c))

        # ---- UPCOMING / OVERDUE EMIs ----
        st.markdown('<div class="payment-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#ffff00!important;text-shadow:0 0 15px #ffaa00;'>📅 EMI Status Board</h3>", unsafe_allow_html=True)

        upcoming_rows = sched_df[sched_df["Status"].isin(["Upcoming","Due Today","Due Soon","Overdue"])].head(6)

        if upcoming_rows.empty:
            st.success("🎉 All EMIs up to date!")
        else:
            for _, row in upcoming_rows.iterrows():
                mo       = int(row["Month"])
                st_cls   = {"Overdue":"overdue-emi","Due Today":"due-emi","Due Soon":"due-emi"}.get(row["Status"], "upcoming-emi")
                late_lbl = f" | 🔴 Late Fee: ₹{row['Late Fee (₹)']}" if row["Late Fee (₹)"] > 0 else ""
                st.markdown(
                    f"<div class='{st_cls}'>Month {mo} — Due: {row['Due Date']} — EMI: ₹{row['EMI (₹)']:,} {late_lbl} — <b>{row['Status']}</b></div>",
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

        # ---- PAYMENT GATEWAY ----
        st.markdown('<div class="gateway-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#66bbff!important;text-shadow:0 0 15px #3399ff;'>🏦 Payment Gateway</h3>", unsafe_allow_html=True)

        unpaid_months = sched_df[~sched_df["Month"].isin(paid_months)]
        unpaid_options = [f"Month {int(r['Month'])} — Due {r['Due Date']} — ₹{r['EMI (₹)']+r['Late Fee (₹)']:,.2f}"
                          for _, r in unpaid_months.iterrows()]

        if unpaid_options:
            sel_emi_lbl = st.selectbox("Select EMI to Pay", unpaid_options)
            sel_emi_mo  = int(sel_emi_lbl.split("Month ")[1].split(" ")[0])
            emi_row     = sched_df[sched_df["Month"] == sel_emi_mo].iloc[0]
            total_due   = emi_row["EMI (₹)"] + emi_row["Late Fee (₹)"]

            st.markdown(f"""
            <div style='background:#001a33;border:1px solid #3399ff55;border-radius:14px;padding:14px 20px;margin:10px 0;'>
            <div style='color:#66bbff;font-size:13px;'>
            📅 Due Date: <b>{emi_row['Due Date']}</b> &nbsp;|&nbsp;
            💰 EMI: <b>₹{emi_row['EMI (₹)']:,}</b> &nbsp;|&nbsp;
            🔴 Late Fee: <b>₹{emi_row['Late Fee (₹)']:,}</b> &nbsp;|&nbsp;
            💳 Total: <b style='color:#00ffe0;font-size:16px;'>₹{total_due:,.2f}</b>
            </div></div>
            """, unsafe_allow_html=True)

            pay_method = st.radio("Payment Method", ["📱 UPI", "💳 Card", "🏦 Net Banking"], horizontal=True)

            if pay_method == "📱 UPI":
                upi_id = st.text_input("UPI ID", placeholder="yourname@upi")
                pay_ready = bool(upi_id and "@" in upi_id)

            elif pay_method == "💳 Card":
                pc1, pc2 = st.columns(2)
                card_num  = pc1.text_input("Card Number", max_chars=16, placeholder="16-digit number")
                card_exp  = pc2.text_input("Expiry (MM/YY)", max_chars=5, placeholder="MM/YY")
                card_cvv  = pc1.text_input("CVV", max_chars=3, type="password")
                card_name = pc2.text_input("Name on Card")
                pay_ready = len(card_num) == 16 and card_name and card_exp and card_cvv

            else:  # Net Banking
                nb_bank = st.selectbox("Select Bank", [
                    "SBI", "HDFC", "ICICI", "Axis", "PNB",
                    "Kotak", "Yes Bank", "IndusInd"
                ])
                nb_uid  = st.text_input("Net Banking User ID")
                nb_pwd  = st.text_input("Password", type="password")
                pay_ready = bool(nb_uid and nb_pwd)

            if st.button("💸 Pay Now", key="pay_now_btn"):
                with st.spinner("Processing payment..."):
                    time.sleep(1.5)

                txn_id = "TXN" + hashlib.md5(
                    (st.session_state.username + str(sel_emi_mo) + str(time.time())).encode()
                ).hexdigest()[:12].upper()

                pay_record = {
                    "Username":       st.session_state.username,
                    "Loan_Index":     sel_idx,
                    "Month":          sel_emi_mo,
                    "Due_Date":       emi_row["Due Date"],
                    "Paid_Date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Principal":      emi_row["Principal (₹)"],
                    "Interest":       emi_row["Interest (₹)"],
                    "EMI_Amount":     emi_row["EMI (₹)"],
                    "Late_Fee":       emi_row["Late Fee (₹)"],
                    "Total_Paid":     total_due,
                    "Payment_Method": pay_method,
                    "Transaction_ID": txn_id,
                    "Status":         "Success"
                }
                cur_pays = load_payments()
                cur_pays = pd.concat([cur_pays, pd.DataFrame([pay_record])], ignore_index=True)
                cur_pays.to_csv(PAYMENTS_FILE, index=False)

                # check if all EMIs paid → close loan
                paid_now = set(cur_pays[(cur_pays["Username"] == st.session_state.username) &
                                        (cur_pays["Loan_Index"] == sel_idx)]["Month"].astype(int).tolist())
                if len(paid_now) >= int(t):
                    loans_df_upd = pd.read_csv(LOANS_FILE)
                    loans_df_upd.at[sel_idx, "Loan Status"]     = "Closed"
                    loans_df_upd.at[sel_idx, "Lifecycle Stage"] = "Loan Closed"
                    loans_df_upd.to_csv(LOANS_FILE, index=False)

                st.session_state.last_receipt = pay_record
                st.success("✅ Payment Successful!")
                st.rerun()
        else:
            st.success("🎉 All EMIs for this loan are paid!")

        st.markdown('</div>', unsafe_allow_html=True)

        # ---- PAYMENT RECEIPT ----
        if st.session_state.last_receipt:
            rc = st.session_state.last_receipt
            st.markdown('<div class="receipt-card">', unsafe_allow_html=True)
            st.markdown("""<h3 style='color:#00ff88!important;text-shadow:0 0 15px #00ff44;
            text-align:center;letter-spacing:3px;'>🧾 PAYMENT RECEIPT</h3>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;color:#00ff88;font-size:13px;'>
            <div>🔢 Transaction ID</div><div><b>{rc['Transaction_ID']}</b></div>
            <div>📅 Paid Date</div><div><b>{rc['Paid_Date']}</b></div>
            <div>📆 EMI Month</div><div><b>{rc['Month']}</b></div>
            <div>💰 EMI Amount</div><div><b>₹ {rc['EMI_Amount']:,}</b></div>
            <div>🔴 Late Fee</div><div><b>₹ {rc['Late_Fee']:,}</b></div>
            <div>💳 Total Paid</div><div><b style='color:#00ffe0;font-size:16px;'>₹ {rc['Total_Paid']:,}</b></div>
            <div>🏦 Method</div><div><b>{rc['Payment_Method']}</b></div>
            <div>✅ Status</div><div><b style='color:#00ff88;'>SUCCESS</b></div>
            </div>
            """, unsafe_allow_html=True)
            rpt_txt = f"""PAYMENT RECEIPT
Transaction ID : {rc['Transaction_ID']}
Paid Date      : {rc['Paid_Date']}
EMI Month      : {rc['Month']}
EMI Amount     : Rs. {rc['EMI_Amount']}
Late Fee       : Rs. {rc['Late_Fee']}
Total Paid     : Rs. {rc['Total_Paid']}
Method         : {rc['Payment_Method']}
Status         : SUCCESS
"""
            st.download_button("⬇ Download Receipt", rpt_txt,
                               file_name=f"receipt_{rc['Transaction_ID']}.txt")
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- PAYMENT HISTORY ----
        st.markdown("### 📜 Payment History")
        pay_df2 = load_payments()
        hist = pay_df2[(pay_df2["Username"] == st.session_state.username) &
                       (pay_df2["Loan_Index"] == sel_idx)]
        if hist.empty:
            st.info("No payments made yet for this loan.")
        else:
            show_cols = ["Month","Due_Date","Paid_Date","EMI_Amount","Late_Fee","Total_Paid","Payment_Method","Transaction_ID","Status"]
            st.dataframe(hist[show_cols].sort_values("Month"), use_container_width=True)

        # ---- FULL EMI SCHEDULE TABLE ----
        st.markdown("### 📅 Full EMI Schedule")
        st.dataframe(sched_df, use_container_width=True, height=320)

    # =====================================================================
    # PAGE: LOAN APPLICATION
    # =====================================================================
    elif menu == "🏠 Loan Application":
        st.title("⚡ bank loan approval prediction")

        # sidebar inputs
        st.sidebar.title("📊 Loan Feature Inputs")
        with st.sidebar.expander("⬇️ Loan Applications", expanded=False):
            feat_inputs = {f: st.number_input(f, value=0.0, key=f"f_{f}") for f in FEATURES}

        st.sidebar.markdown("## 🧠 Advanced Controls")
        if st.sidebar.button("⚡ Open Advanced Panel"):
            st.session_state.show_adv = True

        # personal info row
        st.subheader("👤 Personal Info")
        c1, c2, c3, c4 = st.columns(4)
        p_name   = c1.text_input("Full Name")
        p_age    = c2.number_input("Age", 18, 100, 25)
        p_gender = c3.selectbox("Gender", ["Male", "Female"])
        p_nat    = c4.text_input("Nationality")
        p_mar    = st.selectbox("Marital Status", ["Single", "Married"])
        p_ltype  = st.selectbox("Loan Type", list(LOANS.keys()))
        p_cscore = st.slider("Credit Score", 300, 900, 700)

        # EMI day selector
        st.subheader("📅 EMI Preferences")
        emi_day_sel = st.slider("Preferred EMI Due Day of Month", 1, 28, 5,
                                help="Day of month when EMI will be deducted each month")
        st.info(f"Your EMI will be deducted on day **{emi_day_sel}** of every month.")

        # document uploads
        st.subheader("📄 Documents")
        f_aad  = st.file_uploader("Aadhaar Card",    key="up_a")
        f_pan  = st.file_uploader("PAN Card",        key="up_p")
        f_cred = st.file_uploader("Credit Report",   key="up_c")
        f_bank = st.file_uploader("Bank Statement",  key="up_b")

        # KYC status notice
        kyc_rec = get_user_kyc(st.session_state.username)
        kyc_ok  = kyc_rec is not None and str(kyc_rec.get("KYC_Status","")) == "Verified"
        if kyc_ok:
            st.success(f"✅ KYC Verified — Loan will be credited to {kyc_rec.get('Bank_Name','your bank')} A/c ending {str(kyc_rec.get('Account_Number',''))[-4:]}")
        else:
            st.warning("⚠️ KYC not completed. Complete KYC from the sidebar to enable disbursement.")

        # advanced popup
        if st.session_state.get("show_adv", False):
            st.markdown("## 🧠 Advanced Loan Intelligence Panel")

        with st.container():
            st.info("⚡ Adjust AI-driven loan controls below")
            ac1, ac2 = st.columns(2)
            with ac1:
                adv_smart_emi  = st.checkbox("💡 Smart EMI Optimization")
                adv_dyn_int    = st.checkbox("📉 Dynamic Interest Adjustment")
                adv_auto_appr  = st.checkbox("🤖 Auto Approval Mode")
                adv_stress     = st.checkbox("📊 Income Stress Test")
                adv_restr      = st.checkbox("🔁 Loan Restructure")
            with ac2:
                adv_risk_sens  = st.slider("⚠️ Risk Sensitivity", 0.5, 2.0, 1.0)
                adv_fraud_str  = st.slider("🕵️ Fraud Detection Strength", 0.1, 1.0, 0.5)
                adv_cr_boost   = st.slider("📈 Credit Boost", -50, 100, 0)
            if st.button("❌ Close Panel"):
                st.session_state.show_adv = False

        # submit loan application
        if st.button("🚀 Apply Loan", key="apply_btn"):
            X_in = np.array([list(feat_inputs.values())])
            if rf_model:
                pred = int(rf_model.predict_proba(scaler.transform(X_in))[0][1] > 0.5)
            else:
                pred = 1

            rate_a, ten_a = LOANS[p_ltype]
            emi_a = emi_calc(feat_inputs["Loan Amount"], rate_a, ten_a)
            risk_a = "Low" if p_cscore > 750 else "Medium" if p_cscore > 650 else "High"
            fraud_a = IsolationForest(random_state=42).fit(np.random.rand(50, 4)).predict([X_in[0][:4]])[0] == -1

            appr_prob = round(min(100, max(0,
                (p_cscore - 300) / 600 * 50 +
                min(feat_inputs.get("Applicant Income", 0) / 200000 * 30, 30) +
                (1 - min(feat_inputs.get("DTI Ratio", 0.5), 1)) * 20
            )), 1)

            expl = str(get_explain_scores(feat_inputs, p_cscore, feat_inputs["Loan Amount"]))

            # bank details from KYC
            acc_n  = kyc_rec.get("Account_Number", "") if kyc_ok else ""
            bnk_n  = kyc_rec.get("Bank_Name", "")      if kyc_ok else ""
            ifc_c  = kyc_rec.get("IFSC_Code", "")      if kyc_ok else ""

            new_entry = {f: feat_inputs.get(f, None) for f in FEATURES}
            new_entry.update({
                "Username": st.session_state.username,
                "Name": p_name, "Age": p_age, "Gender": p_gender,
                "Nationality": p_nat, "Marital Status": p_mar,
                "Aadhaar": store_upload(f_aad), "PAN": store_upload(f_pan),
                "CreditFile": store_upload(f_cred), "BankFile": store_upload(f_bank),
                "Prediction": pred, "Risk": risk_a, "Fraud": fraud_a,
                "EMI": emi_a, "Loan Type": p_ltype, "Loan Status": "Under Review",
                "Credit Score": p_cscore,
                "Applied Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Lifecycle Stage": "Application Submitted",
                "Approval Probability": appr_prob,
                "Explainability": expl,
                "EMI_Day": emi_day_sel,
                "Disbursement_Date": datetime.now().strftime("%Y-%m-%d"),
                "Account_Number": acc_n,
                "Bank_Name": bnk_n,
                "IFSC_Code": ifc_c,
            })
            loans_df = pd.concat([loans_df, pd.DataFrame([new_entry])], ignore_index=True)
            loans_df.to_csv(LOANS_FILE, index=False)
            st.success("✅ Loan application submitted!")
            st.rerun()

        # reload after submit
        loans_df = pd.read_csv(LOANS_FILE)
        for c in ALL_COLS:
            if c not in loans_df.columns:
                loans_df[c] = None

        # current loan params
        rate_now, ten_now = LOANS[p_ltype]
        r_now    = rate_now / 12
        l_amt    = feat_inputs["Loan Amount"]
        emi_now  = emi_calc(l_amt, rate_now, ten_now)
        risk_now = "Low" if p_cscore > 750 else "Medium" if p_cscore > 650 else "High"

        # =================== LIFECYCLE TRACKER ===================
        st.markdown('<div class="lifecycle-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#00ffe0!important;text-shadow:0 0 15px #00ffe0;'>🧬 Loan Lifecycle Tracker</h3>",
                    unsafe_allow_html=True)

        cur_status = "Under Review"
        cur_aprob  = 0.0
        cur_date   = "N/A"
        if not loans_df.empty and loans_df["Name"].notna().any():
            last_row   = loans_df.iloc[-1]
            cur_status = str(last_row.get("Loan Status", "Under Review") or "Under Review")
            try:    cur_aprob = float(last_row.get("Approval Probability", 0) or 0)
            except: cur_aprob = 0.0
            cur_date = str(last_row.get("Applied Date", "N/A") or "N/A")

        lc_now = get_stage_idx(cur_status)
        st.markdown(render_lifecycle(lc_now), unsafe_allow_html=True)

        lm1, lm2, lm3 = st.columns(3)
        lm1.metric("📌 Current Stage",        LC_STEPS[lc_now][0])
        lm2.metric("🎯 Approval Probability", f"{cur_aprob:.1f}%")
        lm3.metric("📅 Applied On",           cur_date)
        st.markdown('</div>', unsafe_allow_html=True)

        # =================== AI LIGHT CONE ===================
        st.subheader("🧠 AI Light Cone Decision Engine")

        cone_lbls = ["Digital","Spending","Lifestyle","Social","Discipline",
                    "Income","Risk","Fraud","Eligibility","Approval"]
        cone_scores = np.array([np.random.randint(a,b) for a,b in [
            (60,95),(50,90),(40,85),(55,92),(60,98),(50,100),(5,40),(1,30),(60,95),(65,99)]])

        hc, rc_v = 6, 4
        ta = np.linspace(0, 2*np.pi, 60)
        za = np.linspace(0, hc, 40)
        ta, za = np.meshgrid(ta, za)
        Xc = (za/hc)*rc_v*np.cos(ta)
        Yc = (za/hc)*rc_v*np.sin(ta)
        gx, gy = np.meshgrid(np.linspace(-rc_v, rc_v, 20), np.linspace(-rc_v, rc_v, 20))

        px_pts, py_pts, pz_pts = [], [], []
        for i in range(len(cone_lbls)):
            t_ = i / (len(cone_lbls)-1)
            rad = rc_v*(1-abs(t_-0.5)*1.6)
            ang = t_*4*np.pi
            px_pts.append(np.cos(ang)*rad*0.8)
            py_pts.append(np.sin(ang)*rad*0.8)
            pz_pts.append((t_-0.5)*8)

        fig_cone = go.Figure()
        fig_cone.add_trace(go.Surface(x=Xc, y=Yc, z=za,  colorscale=[[0,"#00eaff"],[1,"#007bff"]], opacity=0.85, showscale=False))
        fig_cone.add_trace(go.Surface(x=Xc, y=Yc, z=-za, colorscale=[[0,"#007bff"],[1,"#8a2be2"]], opacity=0.85, showscale=False))
        fig_cone.add_trace(go.Surface(x=gx, y=gy, z=np.zeros_like(gx), colorscale=[[0,"#00eaff"],[1,"#0047ff"]], opacity=0.3, showscale=False))
        fig_cone.add_trace(go.Scatter3d(
            x=px_pts, y=py_pts, z=pz_pts,
            mode='markers+text', text=cone_lbls, textposition="top center",
            marker=dict(size=[s/8 for s in cone_scores], color=cone_scores, colorscale="Turbo", opacity=0.95)
        ))
        fig_cone.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers+text', text=["⚡ AI CORE"],
            marker=dict(size=14, color="#00eaff"), textposition="bottom center"
        ))
        fig_cone.update_layout(
            title="🔮 AI Light Cone — Neon Decision Engine",
            paper_bgcolor="#001a33",
            scene=dict(bgcolor="#001a33",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False)),
            font=dict(color="#00eaff", family="Orbitron"),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_cone, use_container_width=True)

        ai_call = "APPROVED ✅" if cone_scores[-1] > 70 else "REVIEW ⚠️"
        st.markdown(f"""<div style='text-align:center;font-size:26px;color:#00eaff;padding:18px;
        border-radius:20px;box-shadow:0 0 30px #00eaff;margin:15px 0;
        background:linear-gradient(135deg,#001a33,#002b55);letter-spacing:3px;font-family:Orbitron;'>
        🧠 FINAL AI DECISION: <b>{ai_call}</b></div>""", unsafe_allow_html=True)

        # =================== LOAN SUMMARY CARD ===================
        st.markdown(f"""<div class="loan-card">
            <h3 style='color:#00ffe0!important;text-shadow:0 0 20px #00ffe0;margin-bottom:16px;'>🤖  Loan Approval Card</h3>
            <div style='display:flex;flex-wrap:wrap;justify-content:space-around;gap:14px;'>
                <div style='background:#00ffe011;border:1px solid #00ffe044;border-radius:14px;padding:14px 22px;text-align:center;min-width:130px;'>
                    <div style='font-size:10px;color:#00ffe099;letter-spacing:2px;'>💰 LOAN AMOUNT</div>
                    <div style='font-size:18px;color:#00ffe0;font-weight:bold;'>₹ {l_amt:,.0f}</div>
                </div>
                <div style='background:#ff00aa11;border:1px solid #ff00aa44;border-radius:14px;padding:14px 22px;text-align:center;min-width:130px;'>
                    <div style='font-size:10px;color:#ff88cc;letter-spacing:2px;'>📆 MONTHLY EMI</div>
                    <div style='font-size:18px;color:#ff88cc;font-weight:bold;'>₹ {emi_now:,.2f}</div>
                </div>
                <div style='background:#ffff0011;border:1px solid #ffff0044;border-radius:14px;padding:14px 22px;text-align:center;min-width:130px;'>
                    <div style='font-size:10px;color:#ffff88;letter-spacing:2px;'>📊 CREDIT SCORE</div>
                    <div style='font-size:18px;color:#ffff00;font-weight:bold;'>{p_cscore}</div>
                </div>
                <div style='background:#ff440011;border:1px solid #ff440044;border-radius:14px;padding:14px 22px;text-align:center;min-width:130px;'>
                    <div style='font-size:10px;color:#ff8888;letter-spacing:2px;'>⚠️ RISK LEVEL</div>
                    <div style='font-size:18px;color:#ff6666;font-weight:bold;'>{risk_now}</div>
                </div>
                <div style='background:#00ff4411;border:1px solid #00ff4444;border-radius:14px;padding:14px 22px;text-align:center;min-width:130px;'>
                    <div style='font-size:10px;color:#88ff88;letter-spacing:2px;'>✅ STATUS</div>
                    <div style='font-size:18px;color:#00ff88;font-weight:bold;'>Under Review</div>
                </div>
                <div style='background:#aa00ff11;border:1px solid #aa00ff44;border-radius:14px;padding:14px 22px;text-align:center;min-width:130px;'>
                    <div style='font-size:10px;color:#cc88ff;letter-spacing:2px;'>💳 LOAN TYPE</div>
                    <div style='font-size:16px;color:#cc88ff;font-weight:bold;'>{p_ltype}</div>
                </div>
                <div style='background:#00ff8811;border:1px solid #00ff8844;border-radius:14px;padding:14px 22px;text-align:center;min-width:130px;'>
                    <div style='font-size:10px;color:#88ffcc;letter-spacing:2px;'>📅 EMI DUE DAY</div>
                    <div style='font-size:18px;color:#00ff88;font-weight:bold;'>Day {emi_day_sel}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # =================== EXPLAINABILITY ===================
        st.markdown('<div class="explain-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#ff88ff!important;text-shadow:0 0 15px #ff00cc;'>🔬 AI Explainability — Feature Impact Analysis</h3>",
                    unsafe_allow_html=True)

        expl_data = get_explain_scores(feat_inputs, p_cscore, l_amt)
        expl_keys = list(expl_data.keys())
        expl_vals = list(expl_data.values())
        fig_bar = go.Figure(go.Bar(
                x=expl_vals, y=expl_keys,
                orientation='h',
                marker=dict(
                    color=expl_vals,
                    colorscale=[[0,"#ff3366"],[0.5,"#ffff00"],[1,"#00ff88"]],
                    cmin=0, cmax=1,
                    colorbar=dict(
                        title=dict(text="Impact", font=dict(color="#ff88ff")),
                        tickfont=dict(color="#ff88ff")
                    )
                )
            ))
        fig_bar.update_layout(
            paper_bgcolor="#1a0020", plot_bgcolor="#1a0020",
            font=dict(color="#ff88ff", family="Orbitron"),
            title=dict(text="📊 Feature Contribution to Approval",
                    font=dict(color="#ff88ff", size=13)),
            xaxis=dict(range=[0, 1.2], showgrid=True,
                    gridcolor="rgba(255,0,204,0.1)",
                    tickformat='.0%',
                    tickfont=dict(color="#ff88ff"),
                    zeroline=True,
                    zerolinecolor="rgba(255,0,204,0.3)"),
            yaxis=dict(showgrid=False, tickfont=dict(color="#ff88ff")),
            margin=dict(l=20, r=80, t=50, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="expl_bar")

        fig_rad = go.Figure(go.Scatterpolar(
            r=expl_vals + [expl_vals[0]],
            theta=expl_keys + [expl_keys[0]],
            fill='toself',
            fillcolor='rgba(255,0,204,0.1)',
            line=dict(color="#ff00cc", width=2),
            marker=dict(size=7, color="#ff88ff"),
        ))
        fig_rad.update_layout(
            paper_bgcolor="#1a0020",
            polar=dict(
                bgcolor="#0d0020",
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor="rgba(255,0,204,0.2)",
                                tickfont=dict(color="#ff88ff"),
                                tickformat='.0%'),
                angularaxis=dict(tickfont=dict(color="#ff88ff"),
                                gridcolor="rgba(255,0,204,0.2)"),
            ),
            font=dict(color="#ff88ff", family="Orbitron"),
            title=dict(text="🕸 AI Risk Profile Radar",
                    font=dict(color="#ff88ff", size=13)),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_rad, use_container_width=True, key="expl_radar")

        best  = expl_keys[int(np.argmax(expl_vals))]
        worst = expl_keys[int(np.argmin(expl_vals))]
        st.markdown(f"""<div style='background:#ff00cc11;border:1px solid #ff00cc44;
        border-radius:16px;padding:16px 22px;margin-top:10px;'>
        <div style='color:#ff88ff;font-size:13px;letter-spacing:1px;'>
        🟢 <b>Strongest factor:</b> {best}<br>
        🔴 <b>Weakest factor:</b> {worst}<br>
        🤖 <b>AI Tip:</b> Improving <b>{worst}</b> could boost approval chances by up to 15%.
        </div></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # =================== HEATMAPS ===================
        st.markdown('<div class="heatmap-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#ff8800!important;text-shadow:0 0 15px #ff6600;'>🔥 High-Level Heatmap Intelligence</h3>",
                    unsafe_allow_html=True)

        hc1, hc2 = st.columns(2)

        cs_bands = ["300-450","450-600","600-700","700-800","800-900"]
        la_bands = ["<1L","1-5L","5-15L","15-50L","50L+"]
        risk_mat = np.array([
            [95,85,70,55,40],
            [80,68,52,38,25],
            [60,50,38,28,18],
            [35,28,22,15,10],
            [15,12,10, 8, 5],
        ])
        fig_hm1 = go.Figure(go.Heatmap(
            z=risk_mat, x=la_bands, y=cs_bands,
            colorscale=[[0,"#00ff88"],[0.4,"#ffff00"],[0.75,"#ff6600"],[1,"#ff0033"]],
            text=[[f"{v}%" for v in row] for row in risk_mat],
            texttemplate="%{text}",
            textfont=dict(color="white", family="Orbitron", size=12),
            showscale=True,
            colorbar=dict(title=dict(text="Risk%", font=dict(color="#ff8800")),
                          tickfont=dict(color="#ff8800"))
        ))
        cs_i = min(int((p_cscore - 300) / 120), 4)
        la_i = 0 if l_amt < 100000 else 1 if l_amt < 500000 else 2 if l_amt < 1500000 else 3 if l_amt < 5000000 else 4
        fig_hm1.add_shape(type="rect",
            x0=la_i-0.5, x1=la_i+0.5, y0=cs_i-0.5, y1=cs_i+0.5,
            line=dict(color="#00eaff", width=3))
        fig_hm1.add_annotation(x=la_i, y=cs_i, text="YOU", showarrow=False,
            font=dict(color="#00eaff", size=9, family="Orbitron"), yshift=28)
        fig_hm1.update_layout(
            paper_bgcolor="#1a0800", plot_bgcolor="#1a0800",
            font=dict(color="#ff8800", family="Orbitron"),
            title=dict(text="⚠️ Risk Matrix: Credit Score vs Loan Amount",
                    font=dict(color="#ff8800", size=13)),
            xaxis=dict(title="Loan Amount", tickfont=dict(color="#ff8800"), showgrid=False),
            yaxis=dict(title="Credit Score Band", tickfont=dict(color="#ff8800"), showgrid=False),
            margin=dict(l=20, r=20, t=50, b=40),
        )
        hc1.plotly_chart(fig_hm1, use_container_width=True, key="hm_risk")

        dti_b = ["DTI<20%","20-30%","30-40%","40-50%","50%+"]
        inc_b = ["<20K","20-50K","50-100K","100-200K","200K+"]
        appr_mat = np.array([
            [97,92,88,80,70],
            [88,82,75,65,52],
            [70,62,55,44,32],
            [48,40,33,25,18],
            [22,18,14,10, 7],
        ])
        fig_hm2 = go.Figure(go.Heatmap(
            z=appr_mat, x=inc_b, y=dti_b,
            colorscale=[[0,"#ff0033"],[0.4,"#ff8800"],[0.7,"#ffff00"],[1,"#00ff88"]],
            text=[[f"{v}%" for v in row] for row in appr_mat],
            texttemplate="%{text}",
            textfont=dict(color="white", family="Orbitron", size=12),
            showscale=True,
            colorbar=dict(title=dict(text="Approval%", font=dict(color="#ff8800")),
                          tickfont=dict(color="#ff8800"))
        ))
        fig_hm2.update_layout(
            paper_bgcolor="#1a0800", plot_bgcolor="#1a0800",
            font=dict(color="#ff8800", family="Orbitron"),
            title=dict(text="✅ Approval Probability: DTI vs Income Level",
                    font=dict(color="#ff8800", size=13)),
            xaxis=dict(title="Monthly Income Band", tickfont=dict(color="#ff8800"), showgrid=False),
            yaxis=dict(title="DTI Band", tickfont=dict(color="#ff8800"), showgrid=False),
            margin=dict(l=20, r=20, t=50, b=40),
        )
        hc2.plotly_chart(fig_hm2, use_container_width=True, key="hm_appr")

        hm_cols = [
            "Applicant Income","Coapplicant Income","Loan Amount","Credit History",
            "Total Income","Loan-to-Income Ratio","DTI Ratio","Credit Score"
        ]
        np.random.seed(42)
        raw_data = np.random.randn(200, len(hm_cols))
        raw_data[:,0] *= 50000
        raw_data[:,1] *= 30000
        raw_data[:,2]  = np.abs(raw_data[:,0])*2 + np.random.randn(200)*20000
        raw_data[:,3]  = np.clip(raw_data[:,3]+0.5, 0, 1)
        raw_data[:,4]  = raw_data[:,0] + raw_data[:,1]
        raw_data[:,5]  = raw_data[:,2] / (raw_data[:,4]+1)
        raw_data[:,6]  = raw_data[:,2] / (raw_data[:,4]+1) * 0.5
        raw_data[:,7]  = np.clip(600 + raw_data[:,0]/1000, 300, 900)
        corr_df  = pd.DataFrame(raw_data, columns=hm_cols)
        corr_mat = corr_df.corr().values

        fig_hm3 = go.Figure(go.Heatmap(
            z=corr_mat, x=hm_cols, y=hm_cols,
            colorscale="Viridis",
            colorbar=dict(title=dict(text="Correlation", font=dict(color="#ff8800")),
                          tickfont=dict(color="#ff8800"))
        ))
        fig_hm3.update_layout(
            paper_bgcolor="#1a0800", plot_bgcolor="#1a0800",
            font=dict(color="#ff8800", family="Orbitron"),
            title=dict(text="🔗 Financial Feature Correlation Matrix",
                    font=dict(color="#ff8800", size=13)),
            xaxis=dict(tickangle=-35, tickfont=dict(color="#ff8800", size=9), showgrid=False),
            yaxis=dict(tickfont=dict(color="#ff8800", size=9), showgrid=False),
            margin=dict(l=20, r=20, t=50, b=80),
            height=420,
        )
        st.plotly_chart(fig_hm3, use_container_width=True, key="hm_corr")
        st.markdown('</div>', unsafe_allow_html=True)

        # =================== EMI SCHEDULE ===================
        st.markdown('<div class="emi-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#df80ff!important;text-shadow:0 0 15px #bf00ff;'>📅 EMI Repayment Schedule</h3>",
                    unsafe_allow_html=True)

        bal_run = l_amt
        emi_rows = []
        due_dates_now = get_emi_due_dates(datetime.now().strftime("%Y-%m-%d"), emi_day_sel, int(ten_now))
        for mo in range(1, int(ten_now)+1):
            int_part = bal_run * r_now
            pri_part = emi_now - int_part
            bal_run -= pri_part
            due = due_dates_now[mo-1]
            emi_rows.append({
                "Month":         mo,
                "Due Date":      due.strftime("%Y-%m-%d"),
                "EMI (₹)":       round(emi_now, 2),
                "Principal (₹)": round(pri_part, 2),
                "Interest (₹)":  round(int_part, 2),
                "Balance (₹)":   round(max(bal_run, 0), 2)
            })
        sched_df2 = pd.DataFrame(emi_rows)
        st.dataframe(sched_df2, use_container_width=True, height=320)
        st.markdown('</div>', unsafe_allow_html=True)

        # =================== 6 ANALYTICS CHARTS ===================
        st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#5599ff!important;text-shadow:0 0 15px #0066ff;'>📊 Loan Analytics — 6 Chart Intelligence Panel</h3>",
                    unsafe_allow_html=True)

        if loans_df.empty or loans_df["Loan Type"].dropna().empty:
            demo = []
            for lt, (rr, tt) in LOANS.items():
                for _ in range(np.random.randint(2, 6)):
                    cs = np.random.randint(400, 900)
                    demo.append({
                        "Loan Type":           lt,
                        "Risk":                np.random.choice(["Low","Medium","High"]),
                        "Total Income":        np.random.randint(20000, 200000),
                        "Loan Amount":         np.random.randint(50000, 2000000),
                        "Credit Score":        cs,
                        "Applicant Income":    np.random.randint(15000, 150000),
                        "EMI":                 emi_calc(np.random.randint(50000, 500000), rr, tt),
                        "DTI Ratio":           round(np.random.uniform(0.1, 0.6), 2),
                        "Loan-to-Income Ratio":round(np.random.uniform(0.5, 5.0), 2),
                    })
            chart_df = pd.DataFrame(demo)
        else:
            chart_df = loans_df.copy()

        CBG = "#000a1e"
        CTC = "#5599ff"

        r1c1, r1c2, r1c3 = st.columns(3)

        fig_pie = px.pie(chart_df, names="Loan Type", title="🍩 Loan Type Distribution",
                        color_discrete_sequence=px.colors.sequential.Rainbow, hole=0.45)
        fig_pie.update_layout(paper_bgcolor=CBG, plot_bgcolor=CBG,
                            font=dict(color=CTC, family="Orbitron"),
                            title_font=dict(color=CTC),
                            legend=dict(font=dict(color=CTC), bgcolor="rgba(0,0,0,0)"),
                            margin=dict(l=10,r=10,t=40,b=10))
        r1c1.plotly_chart(fig_pie, use_container_width=True, key="c1")

        fig_hist = px.histogram(chart_df, x="Risk", color="Risk", title="⚡ Risk Distribution",
                                color_discrete_map={"Low":"#00ff88","Medium":"#ffaa00","High":"#ff3366"})
        fig_hist.update_layout(**chart_style("⚡ Risk Distribution", CBG, CTC))
        r1c2.plotly_chart(fig_hist, use_container_width=True, key="c2")

        fig_sct = px.scatter(chart_df, x="Total Income", y="Loan Amount", color="Credit Score",
                            title="💎 Loan vs Income vs Credit", color_continuous_scale="Turbo")
        fig_sct.update_layout(**chart_style("💎 Loan vs Income vs Credit", CBG, CTC))
        r1c3.plotly_chart(fig_sct, use_container_width=True, key="c3")

        r2c1, r2c2, r2c3 = st.columns(3)

        fig_bar2 = px.bar(chart_df, x="Loan Type", color="Loan Type", title="📦 Loan Type Count",
                        color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_bar2.update_layout(**chart_style("📦 Loan Type Count", CBG, CTC))
        fig_bar2.update_xaxes(tickangle=-45)
        r2c1.plotly_chart(fig_bar2, use_container_width=True, key="c4")

        fig_box = px.box(chart_df, y="Credit Score", color="Loan Type", title="📐 Credit Score Spread")
        fig_box.update_layout(**chart_style("📐 Credit Score Spread", CBG, CTC))
        r2c2.plotly_chart(fig_box, use_container_width=True, key="c5")

        num_df = chart_df.select_dtypes(include=np.number).corr()
        fig_cor = px.imshow(num_df, title="🔥 Feature Correlation",
                            color_continuous_scale="Plasma", text_auto=".1f")
        fig_cor.update_layout(paper_bgcolor=CBG, plot_bgcolor=CBG,
                            font=dict(color=CTC, family="Orbitron"),
                            title_font=dict(color=CTC),
                            margin=dict(l=10,r=10,t=40,b=10))
        r2c3.plotly_chart(fig_cor, use_container_width=True, key="c6")
        st.markdown('</div>', unsafe_allow_html=True)

        # =================== ZIGZAG DASHBOARD ===================
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#00ff88!important;text-shadow:0 0 15px #00ff44;'>🚀 Loan Dashboard — Futuristic Multi-ZigZag Intelligence</h3>",
                    unsafe_allow_html=True)

        mths = list(range(1, int(ten_now)+1))
        n    = len(mths)

        bal_t = l_amt
        bals, pris, ints, cum = [], [], [], []
        cpaid = 0
        for _ in mths:
            i_t   = bal_t * r_now
            p_t   = emi_now - i_t
            bal_t -= p_t
            cpaid += emi_now
            bals.append(max(bal_t, 0))
            pris.append(round(p_t, 2))
            ints.append(round(i_t, 2))
            cum.append(round(cpaid, 2))

        zz       = np.array([(-1)**i * np.random.uniform(0, l_amt*0.025) for i in range(n)])
        cr_trend = np.clip(p_cscore + np.cumsum(np.random.randn(n)*4), 300, 900)
        emi_var  = np.array([emi_now + np.random.uniform(-emi_now*0.04, emi_now*0.04) for _ in mths])
        dti_tr   = np.clip(0.35 + np.cumsum(np.random.randn(n)*0.007), 0.05, 0.9)*100
        appr_tr  = np.clip(50 + (cr_trend-500)/10 + np.random.randn(n)*2, 0, 100)

        DBG = "#000f00"

        fig_z1 = go.Figure()
        fig_z1.add_trace(go.Scatter(
            x=mths, y=np.array(bals)+zz,
            mode='lines+markers', name='🔵 Outstanding Balance',
            line=dict(color='#00eaff', width=2.5),
            marker=dict(size=4, symbol='circle', color='#00eaff'),
            fill='tozeroy', fillcolor='rgba(0,234,255,0.05)'
        ))
        fig_z1.add_trace(go.Scatter(
            x=mths, y=np.array(pris)+zz*0.4,
            mode='lines+markers', name='🟢 Principal Per Month',
            line=dict(color='#00ff88', width=2.5, dash='dot'),
            marker=dict(size=5, symbol='diamond', color='#00ff88')
        ))
        fig_z1.add_trace(go.Scatter(
            x=mths, y=np.array(ints),
            mode='lines+markers', name='🔴 Interest Per Month',
            line=dict(color='#ff3366', width=2.5, dash='dash'),
            marker=dict(size=5, symbol='triangle-up', color='#ff3366')
        ))
        fig_z1.add_trace(go.Scatter(
            x=mths, y=cum,
            mode='lines', name='🟡 Cumulative Paid',
            line=dict(color='#ffff00', width=2, dash='longdashdot'),
            fill='tonexty', fillcolor='rgba(255,255,0,0.02)'
        ))
        fig_z1.update_layout(
            title=dict(text="📈 ZigZag Chart 1 — Balance · Principal · Interest · Total Paid",
                    font=dict(color="#00ff88", size=13, family="Orbitron")),
            plot_bgcolor=DBG, paper_bgcolor=DBG,
            font=dict(color="#00ff88", family="Orbitron"),
            xaxis=dict(title="Month", showgrid=True, gridcolor="rgba(0,255,100,0.08)",
                    tickfont=dict(color="#00ff88"), zeroline=False),
            yaxis=dict(title="Amount (₹)", showgrid=True, gridcolor="rgba(0,255,100,0.08)",
                    tickfont=dict(color="#00ff88"), zeroline=False),
            legend=dict(bgcolor="rgba(0,0,0,0.6)", bordercolor="rgba(255,0,255,0.2)",
                        borderwidth=1, font=dict(color="#00ff88", size=10)),
            hovermode="x unified",
            margin=dict(l=50, r=20, t=60, b=50)
        )
        if bals:
            fig_z1.add_annotation(
                x=mths[n//2], y=max(bals)*0.88,
                text="⚡ AI Tracked Amortisation", showarrow=False,
                font=dict(color="#00ff88", size=10, family="Orbitron"),
                bgcolor="rgba(0,255,100,0.1)",
                bordercolor="rgba(255,0,255,0.2)", borderwidth=1)
        st.plotly_chart(fig_z1, use_container_width=True, key="zz1")

        fig_z2 = go.Figure()
        fig_z2.add_trace(go.Scatter(
            x=mths, y=cr_trend,
            mode='lines+markers', name='🟣 Credit Score Trend',
            line=dict(color='#bf00ff', width=3),
            marker=dict(size=5, symbol='square', color='#bf00ff'),
            fill='tozeroy', fillcolor='rgba(191,0,255,0.04)'
        ))
        fig_z2.add_trace(go.Scatter(
            x=mths, y=emi_var,
            mode='lines+markers', name='🟠 EMI Variance',
            line=dict(color='#ff6600', width=2.5, dash='dot'),
            marker=dict(size=5, symbol='pentagon', color='#ff6600')
        ))
        fig_z2.add_trace(go.Scatter(
            x=mths, y=dti_tr,
            mode='lines+markers', name='🩵 DTI Ratio (%)',
            line=dict(color='#00ccff', width=2.5, dash='dash'),
            marker=dict(size=5, symbol='triangle-down', color='#00ccff'),
            yaxis='y2'
        ))
        fig_z2.add_trace(go.Scatter(
            x=mths, y=appr_tr,
            mode='lines', name='✅ Approval Probability (%)',
            line=dict(color='#ffff00', width=2, dash='longdash'),
            fill='tonexty', fillcolor='rgba(255,255,0,0.02)',
            yaxis='y2'
        ))
        fig_z2.update_layout(
            title=dict(text="📉 ZigZag Chart 2 — Credit · EMI Variance · DTI · Approval Probability",
                    font=dict(color="#ff88ff", size=13, family="Orbitron")),
            plot_bgcolor=DBG, paper_bgcolor=DBG,
            font=dict(color="#ff88ff", family="Orbitron"),
            xaxis=dict(title="Month", showgrid=True, gridcolor="rgba(255,0,255,0.08)",
                    tickfont=dict(color="#ff88ff"), zeroline=False),
            yaxis=dict(title="Credit Score / EMI (₹)", showgrid=True,
                    gridcolor="rgba(255,0,255,0.08)",
                    tickfont=dict(color="#ff88ff"), zeroline=False),
            yaxis2=dict(title="DTI % / Approval %", overlaying='y', side='right',
                        showgrid=False, tickfont=dict(color="#00ccff"), zeroline=False),
            legend=dict(bgcolor="rgba(0,0,0,0.6)", bordercolor="rgba(255,0,255,0.2)",
                        borderwidth=1, font=dict(color="#ff88ff", size=10)),
            hovermode="x unified",
            margin=dict(l=50, r=70, t=60, b=50)
        )
        fig_z2.add_annotation(
            x=mths[n//3], y=max(cr_trend)*0.95,
            text="🔮 AI Risk Engine Active", showarrow=False,
            font=dict(color="#ff88ff", size=10, family="Orbitron"),
            bgcolor="rgba(191,0,255,0.1)",
            bordercolor="rgba(255,0,255,0.2)", borderwidth=1)
        st.plotly_chart(fig_z2, use_container_width=True, key="zz2")
        st.markdown('</div>', unsafe_allow_html=True)

        # =================== ADVANCED AI PANEL ===================
        st.markdown("## 🧠 Advanced AI Control Panel")

        if st.button("🚀 Open / Close Advanced Panel", key="adv_toggle"):
            st.session_state.show_adv = not st.session_state.show_adv

        if st.session_state.show_adv:
            with st.container():
                st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
                st.markdown("### ⚡ Advanced AI Features Panel")

                st.markdown("### 📊 Credit Intelligence")
                cb = st.selectbox("Credit Bureau", ["TransUnion","Equifax","Experian"], key="cb")
                iv = st.checkbox("Income Verified", key="iv")
                gr = st.slider("Geo Risk Score", 0, 100, 50, key="gr")
                st.markdown("---")

                st.markdown("### 🤖 AI Risk Engine")
                rs  = st.slider("Risk Score",           0, 100, 40, key="rs")
                fs  = st.slider("Fraud Score",          0, 100, 10, key="fs")
                ap  = st.slider("Approval Probability", 0, 100, 70, key="ap")
                st.markdown("---")

                st.markdown("### 🔐 Security & Behavior")
                dt  = st.slider("Device Trust",      0, 100, 80, key="dt")
                lc  = st.slider("Login Consistency", 0, 100, 75, key="lc")
                adn = st.checkbox("Enable Anomaly Detection", key="adn")
                st.markdown("---")

                st.markdown("### 💰 Financial Engine")
                ir  = st.slider("Dynamic Interest Rate", 1, 20, 10, key="ir")
                eo  = st.checkbox("Smart EMI Optimization", key="eo")
                stt = st.checkbox("Run Stress Test",        key="stt")
                st.markdown("---")

                st.markdown("### 📈 Automation")
                arc = st.checkbox("Auto Re-evaluation",   key="arc")
                mon = st.checkbox("Real-time Monitoring", key="mon")
                rec = st.checkbox("AI Recommendations",   key="rec")
                st.markdown("---")

                st.markdown("### 📄 Document AI")
                ocr = st.checkbox("OCR Verification", key="ocr")
                ds  = st.selectbox("Document Status", ["Pending","Verified","Rejected"], key="ds")
                st.markdown("---")

                st.markdown("### 🧠 Final AI Decision")
                fscore = rs*0.3 + (100-fs)*0.2 + ap*0.3 + dt*0.2
                st.metric("🎯 Final AI Score", round(fscore, 2))
                if fscore > 70:
                    st.success("✅ HIGH APPROVAL CHANCE")
                elif fscore > 40:
                    st.warning("⚠️ MEDIUM RISK — FURTHER REVIEW NEEDED")
                else:
                    st.error("❌ HIGH RISK — LOAN NOT RECOMMENDED")

                st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================================
    # PAGE: LOAN DETAILS (REAL BANK VIEW)
    # =====================================================================
    elif menu == "📄 Loan Details (Real Bank View)":
        st.markdown("""
        <style>
        .cyber-header {
            background: linear-gradient(135deg,#050d1a,#0a1f44);
            padding:25px; border-radius:20px; color:#00ffe0;
            box-shadow:0 0 20px rgba(0,255,224,0.3);
        }
        .cyber-metric {
            background: rgba(0,255,224,0.05);
            border:1px solid rgba(0,255,224,0.2);
            border-radius:15px; padding:10px; text-align:center;
            box-shadow:0 0 10px #00ffe0;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<h1 style='color:#00ffe0'>🏦 Loan Dashboard — Cyber Banking View</h1>", unsafe_allow_html=True)

        loans_df = pd.read_csv(LOANS_FILE)
        for c in ALL_COLS:
            if c not in loans_df.columns:
                loans_df[c] = None

        user_loans = loans_df[loans_df["Username"] == st.session_state.username]

        if user_loans.empty:
            st.warning("No loan applications found.")
            st.stop()

        latest = user_loans.iloc[-1]

        # KYC status
        kyc_rec2 = get_user_kyc(st.session_state.username)
        if kyc_rec2 is not None and str(kyc_rec2.get("KYC_Status","")) == "Verified":
            st.success(f"✅ KYC Verified | Bank: {kyc_rec2.get('Bank_Name','N/A')} | A/c: {'*'*8+str(kyc_rec2.get('Account_Number',''))[-4:]}")
        else:
            st.warning("⚠️ KYC Pending — Go to KYC Verification to complete it.")

        # header
        st.markdown(f"""
        <div class="cyber-header">
        <h2>👤 {latest['Name']}</h2>
        <p>Loan Type: {latest['Loan Type']} | Status: <b>{latest['Loan Status']}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # main metrics
        c1, c2, c3, c4 = st.columns(4)
        try:
            la_v = f"₹ {float(latest['Loan Amount']):,}"
        except:
            la_v = str(latest['Loan Amount'])
        try:
            emi_v = f"₹ {float(latest['EMI']):,}"
        except:
            emi_v = str(latest['EMI'])

        c1.markdown(f"<div class='cyber-metric'>💰 Loan<br>{la_v}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='cyber-metric'>📆 EMI<br>{emi_v}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='cyber-metric'>📊 Score<br>{latest['Credit Score']}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='cyber-metric'>🎯 Approval<br>{latest['Approval Probability']}%</div>", unsafe_allow_html=True)

        # bank account info
        if latest.get("Account_Number"):
            st.markdown(f"""
            <div style='background:#001a33;border:1px solid #3399ff55;border-radius:14px;padding:14px 20px;margin:10px 0;'>
            <div style='color:#66bbff;font-size:13px;'>
            🏦 <b>Disbursement Bank:</b> {latest.get('Bank_Name','N/A')} &nbsp;|&nbsp;
            💳 <b>A/c:</b> {'*'*8+str(latest.get('Account_Number',''))[-4:]} &nbsp;|&nbsp;
            🔑 <b>IFSC:</b> {latest.get('IFSC_Code','N/A')}
            </div></div>
            """, unsafe_allow_html=True)

        # AI decision
        st.markdown("## 🤖 AI Decision")
        st.info(f"Risk Level: {latest['Risk']}")
        st.info(f"Fraud Check: {'⚠️ Risk' if latest['Fraud'] else 'Safe ✅'}")

        # timeline
        st.markdown("## 🧬 Loan Timeline")
        try:
            progress = int(float(latest["Approval Probability"]))
        except:
            progress = 0
        st.progress(progress)

        steps = ["Application", "Verification", "Credit Check", "Risk Analysis", "Approval"]
        for i, step in enumerate(steps):
            if i < progress // 20:
                st.success(f"✔ {step}")
            else:
                st.info(f"⏳ {step}")

        # EMI schedule
        st.markdown("## 💰 EMI Schedule")
        try:
            loan_amt = float(latest["Loan Amount"])
            emi_v2   = float(latest["EMI"])
        except:
            loan_amt = 0.0
            emi_v2   = 0.0

        lt2 = str(latest.get("Loan Type","Personal Loan"))
        rate2, t2 = LOANS.get(lt2, (0.10, 12))
        emi_day2  = int(latest.get("EMI_Day", 5) or 5)
        disb2     = str(latest.get("Disbursement_Date", latest.get("Applied Date", datetime.now().strftime("%Y-%m-%d"))) or datetime.now().strftime("%Y-%m-%d"))
        due_dates2 = get_emi_due_dates(disb2, emi_day2, int(t2))

        balance2  = loan_amt
        r2_mo     = rate2 / 12
        schedule2 = []
        for m in range(1, min(int(t2), 13)+1):
            interest2  = balance2 * r2_mo
            principal2 = emi_v2 - interest2
            balance2  -= principal2
            due2 = due_dates2[m-1]
            schedule2.append([m, due2.strftime("%Y-%m-%d"),
                               round(principal2,2), round(interest2,2), round(max(balance2,0),2)])

        df_schedule = pd.DataFrame(schedule2, columns=["Month","Due Date","Principal","Interest","Balance"])
        st.dataframe(df_schedule, use_container_width=True)

        # credit score gauge
        st.markdown("## 📊 Credit Score")
        try:
            cscore_v = float(latest["Credit Score"])
        except:
            cscore_v = 600.0

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cscore_v,
            title={'text': "Score"},
            gauge={
                'axis': {'range': [300, 900], 'tickcolor': "#00ffe0"},
                'bar': {'color': "#00ffe0"},
                'steps': [
                    {'range': [300, 600], 'color': "#330000"},
                    {'range': [600, 750], 'color': "#663300"},
                    {'range': [750, 900], 'color': "#003322"}
                ]
            }
        ))
        st.plotly_chart(cyber_theme(fig_gauge), use_container_width=True)

        # charts
        st.markdown("## 📈 Financial Analytics (Advanced)")

        st.markdown("### 💰 EMI Breakdown (Principal vs Interest)")
        fig_bar_b = go.Figure()
        fig_bar_b.add_bar(name="Principal", x=df_schedule["Month"],
                          y=df_schedule["Principal"], marker=dict(color="#00ffe0"))
        fig_bar_b.add_bar(name="Interest",  x=df_schedule["Month"],
                          y=df_schedule["Interest"],  marker=dict(color="#ff007f"))
        fig_bar_b.update_layout(barmode='stack', title="Monthly EMI Split")
        st.plotly_chart(cyber_theme(fig_bar_b), use_container_width=True)

        st.markdown("### 📉 Loan Balance Trend")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_schedule["Month"], y=df_schedule["Balance"],
            mode='lines+markers', name="Remaining Balance",
            line=dict(color="#00ffe0", width=3), marker=dict(size=6)
        ))
        fig_line.update_layout(title="Loan Balance Over Time")
        st.plotly_chart(cyber_theme(fig_line), use_container_width=True)

        st.markdown("### ⚠️ Risk Component Analysis")
        try:
            app_inc_v = float(latest["Applicant Income"])
        except:
            app_inc_v = 0.0

        risk_values = [
            cscore_v / 900 * 100,
            min(app_inc_v / 200000 * 100, 100),
            float(latest["Approval Probability"] or 0)
        ]
        risk_labels = ["Credit Strength", "Income Strength", "Approval"]
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Bar(
            x=risk_labels, y=risk_values,
            marker=dict(color=["#00ffe0","#ffaa00","#ff007f"]),
            text=[f"{v:.1f}%" for v in risk_values], textposition='outside'
        ))
        fig_risk.update_layout(title="Risk Component Scores")
        st.plotly_chart(cyber_theme(fig_risk), use_container_width=True)

        st.markdown("### 💼 Income vs Loan Burden")
        fig_compare = go.Figure()
        fig_compare.add_bar(
            x=["Income", "EMI"],
            y=[app_inc_v, emi_v2],
            marker=dict(color=["#00ffe0","#ff007f"])
        )
        fig_compare.update_layout(title="Income vs EMI Comparison")
        st.plotly_chart(cyber_theme(fig_compare), use_container_width=True)

        # loan simulator
        st.markdown("## 🧠 Loan Simulator")
        sim_income = st.slider("Adjust Income", 10000, 200000, int(app_inc_v) if app_inc_v else 50000)
        sim_score  = st.slider("Adjust Credit Score", 300, 900, int(cscore_v))
        sim_prob = min(100, (sim_score - 300)/600*50 + (sim_income/200000)*50)
        st.progress(int(sim_prob))
        st.write(f"⚡ New Approval Probability: {sim_prob:.1f}%")

        # payment simulation
        st.markdown("## 💳 Payment Simulation")
        pay_sim = st.number_input("Enter Payment Amount", 0, int(loan_amt) if loan_amt else 1000000, 0)
        remaining_sim = loan_amt - pay_sim
        st.markdown(f"<div class='cyber-metric'>Remaining Balance: ₹ {remaining_sim:,.2f}</div>", unsafe_allow_html=True)

        # download report
        st.markdown("## 📄 Loan Report")
        kyc_info = ""
        if kyc_rec2 is not None:
            kyc_info = f"""
KYC Status     : {kyc_rec2.get('KYC_Status','N/A')}
Phone          : {kyc_rec2.get('Phone','N/A')}
Bank           : {kyc_rec2.get('Bank_Name','N/A')}
IFSC           : {kyc_rec2.get('IFSC_Code','N/A')}
Account        : {'*'*8+str(kyc_rec2.get('Account_Number',''))[-4:]}"""

        report = f"""
LOAN REPORT
===========
Name           : {latest['Name']}
Loan Type      : {latest['Loan Type']}
Loan Amount    : {latest['Loan Amount']}
EMI            : {latest['EMI']}
EMI Due Day    : {latest.get('EMI_Day','N/A')}
Risk           : {latest['Risk']}
Credit Score   : {latest['Credit Score']}
Approval       : {latest['Approval Probability']}%
Status         : {latest['Loan Status']}
Applied Date   : {latest['Applied Date']}
{kyc_info}
"""
        st.download_button("⬇ Download Report", report, file_name="loan_report.txt")

        # payment summary
        st.markdown("## 💳 Payment Summary")
        pay_df3 = load_payments()
        u_pays  = pay_df3[pay_df3["Username"] == st.session_state.username]
        if not u_pays.empty:
            total_paid_amt = u_pays["Total_Paid"].astype(float).sum()
            st.metric("💰 Total Amount Paid", f"₹ {total_paid_amt:,.2f}")
            st.dataframe(u_pays[["Month","Paid_Date","Total_Paid","Payment_Method","Transaction_ID","Status"]],
                         use_container_width=True)
        else:
            st.info("No payments recorded yet. Go to EMI Payment Center to make payments.")

        # final decision
        st.markdown("---")
        try:
            ap_v = float(latest["Approval Probability"])
        except:
            ap_v = 0.0

        if ap_v > 70:
            st.success("✅ LOAN APPROVED")
        elif ap_v > 40:
            st.warning("⚠️ UNDER REVIEW")
        else:
            st.error("❌ LOAN REJECTED")

# ===================== ADMIN SECTION =====================
if st.session_state.login and st.session_state.role == "admin":
    st.title("🛠 Admin Dashboard")

    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["📋 Loan Applications", "🔐 KYC Records", "💳 Payment Records"])

    with admin_tab1:
        loans_df = pd.read_csv(LOANS_FILE)
        for c in ALL_COLS:
            if c not in loans_df.columns:
                loans_df[c] = None

        st.dataframe(loans_df, use_container_width=True)
        st.subheader("📄 User Documents")

        show = [c for c in ["Name","Username","Aadhaar","PAN","CreditFile","BankFile",
                             "Loan Status","Lifecycle Stage","Approval Probability",
                             "Account_Number","Bank_Name","IFSC_Code","EMI_Day"]
                if c in loans_df.columns]
        st.write(loans_df[show])

        for i in loans_df.index:
            a1, a2, a3 = st.columns(3)
            if a1.button(f"✅ Approve #{i}", key=f"apr_{i}"):
                loans_df.at[i, "Loan Status"]     = "Approved"
                loans_df.at[i, "Lifecycle Stage"] = "Final Approval"
                loans_df.at[i, "Last Updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")
                loans_df.to_csv(LOANS_FILE, index=False)
                st.success(f"Application #{i} approved ✅")
                st.rerun()
            if a2.button(f"❌ Reject #{i}", key=f"rej_{i}"):
                loans_df.at[i, "Loan Status"]     = "Rejected"
                loans_df.at[i, "Lifecycle Stage"] = "Credit Bureau Check"
                loans_df.at[i, "Last Updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")
                loans_df.to_csv(LOANS_FILE, index=False)
                st.error(f"Application #{i} rejected ❌")
                st.rerun()
            if a3.button(f"💰 Disburse #{i}", key=f"dis_{i}"):
                loans_df.at[i, "Loan Status"]        = "Disbursed"
                loans_df.at[i, "Lifecycle Stage"]    = "Loan Disbursement"
                loans_df.at[i, "Disbursement_Date"]  = datetime.now().strftime("%Y-%m-%d")
                loans_df.at[i, "Last Updated"]       = datetime.now().strftime("%Y-%m-%d %H:%M")
                loans_df.to_csv(LOANS_FILE, index=False)
                st.success(f"Application #{i} disbursed 💰")
                st.rerun()

    with admin_tab2:
        st.subheader("🔐 KYC Verification Records")
        kyc_df_admin = load_kyc()
        if kyc_df_admin.empty:
            st.info("No KYC records yet.")
        else:
            st.dataframe(kyc_df_admin[["Username","Phone","OTP_Verified","Bank_Name",
                                        "IFSC_Code","KYC_Status","Verified_At"]],
                         use_container_width=True)

    with admin_tab3:
        st.subheader("💳 All Payment Records")
        pay_df_admin = load_payments()
        if pay_df_admin.empty:
            st.info("No payment records yet.")
        else:
            st.dataframe(pay_df_admin, use_container_width=True)
            total_col = pay_df_admin["Total_Paid"].astype(float)
            st.metric("💰 Total Collections", f"₹ {total_col.sum():,.2f}")
            st.metric("✅ Successful Transactions", str(len(pay_df_admin[pay_df_admin["Status"]=="Success"])))
