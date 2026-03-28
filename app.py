import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import bcrypt
import hashlib
import time
from datetime import datetime
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
</style>
""", unsafe_allow_html=True)

# folder setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPL_DIR  = os.path.join(BASE_DIR, "uploads")
MDL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPL_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.csv")
LOANS_FILE = os.path.join(DATA_DIR, "loan.csv")

# create default users if not exist
if not os.path.exists(USERS_FILE):
    default_users = [
        {"username": "admin", "password_hash": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(), "role": "admin"},
        {"username": "user",  "password_hash": bcrypt.hashpw("user123".encode(),  bcrypt.gensalt()).decode(), "role": "user"}
    ]
    pd.DataFrame(default_users).to_csv(USERS_FILE, index=False)

users_df = pd.read_csv(USERS_FILE)

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
    "Name", "Age", "Gender", "Nationality", "Marital Status",
    "Aadhaar", "PAN", "CreditFile", "BankFile",
    "Prediction", "Risk", "Fraud", "EMI", "Loan Type", "Loan Status",
    "Credit Score", "Applied Date", "Last Updated",
    "Lifecycle Stage", "Approval Probability", "Explainability"
]

if not os.path.exists(LOANS_FILE):
    pd.DataFrame(columns=ALL_COLS).to_csv(LOANS_FILE, index=False)

loans_df = pd.read_csv(LOANS_FILE)
for c in ALL_COLS:
    if c not in loans_df.columns:
        loans_df[c] = None

# session defaults
for k, v in [("login", False), ("role", ""), ("username", ""), ("show_adv", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# helper - check login
def check_login(u, p):
    if u in users_df["username"].values:
        row = users_df[users_df["username"] == u].iloc[0]
        if bcrypt.checkpw(p.encode(), row["password_hash"].encode()):
            return True, row["role"]
    return False, None

# helper - register new user
def add_user(u, p):
    if u in users_df["username"].values:
        return False
    h = bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
    pd.DataFrame([{"username": u, "password_hash": h, "role": "user"}]).to_csv(
        USERS_FILE, mode="a", header=False, index=False)
    return True

# helper - save uploaded file to disk
def store_upload(f):
    if f is None:
        return ""
    uid = hashlib.md5((f.name + str(time.time())).encode()).hexdigest()
    dest = os.path.join(UPL_DIR, uid + "_" + f.name)
    with open(dest, "wb") as out:
        out.write(f.getbuffer())
    return uid + "_" + f.name

# emi calc
def emi_calc(principal, annual_rate, months):
    r = annual_rate / 12
    if r == 0:
        return round(principal / months, 2)
    return round(principal * r * (1+r)**months / ((1+r)**months - 1), 2)

# chart layout helper
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

# lifecycle stage list
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

# explainability scores based on inputs
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
        nu = st.text_input("Pick a Username")
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
    "📄 Loan Details (Real Bank View)"
        ])
    if menu == "🏠 Loan Application":

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

        # document uploads
        st.subheader("📄 Documents")
        f_aad  = st.file_uploader("Aadhaar Card",    key="up_a")
        f_pan  = st.file_uploader("PAN Card",        key="up_p")
        f_cred = st.file_uploader("Credit Report",   key="up_c")
        f_bank = st.file_uploader("Bank Statement",  key="up_b")

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
                "Last Updated":  datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Lifecycle Stage": "Application Submitted",
                "Approval Probability": appr_prob,
                "Explainability": expl
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

        hc, rc = 6, 4
        ta = np.linspace(0, 2*np.pi, 60)
        za = np.linspace(0, hc, 40)
        ta, za = np.meshgrid(ta, za)
        Xc = (za/hc)*rc*np.cos(ta)
        Yc = (za/hc)*rc*np.sin(ta)
        gx, gy = np.meshgrid(np.linspace(-rc, rc, 20), np.linspace(-rc, rc, 20))

        px_pts, py_pts, pz_pts = [], [], []
        for i in range(len(cone_lbls)):
            t = i / (len(cone_lbls)-1)
            rad = rc*(1-abs(t-0.5)*1.6)
            ang = t*4*np.pi
            px_pts.append(np.cos(ang)*rad*0.8)
            py_pts.append(np.sin(ang)*rad*0.8)
            pz_pts.append((t-0.5)*8)

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
                        title=dict(
                            text="Impact",
                            font=dict(color="#ff88ff")
                        ),
                        tickfont=dict(color="#ff88ff")
                    )
                )
            ))
            
        text=[f"{v*100:.1f}%" for v in expl_vals],
        textposition='outside',
        textfont=dict(color="#ff88ff"),
        
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

        # radar
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

        # risk matrix heatmap
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
        colorbar=dict(
        title=dict(
            text="Something",
            font=dict(color="#ff8800")
        ),
        tickfont=dict(color="#ff8800")
        )
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

        # approval probability heatmap
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
        colorbar=dict(
        title=dict(
            text="Something",
            font=dict(color="#ff8800")
        ),
        tickfont=dict(color="#ff8800")
        )
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

        # correlation heatmap full width
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
        z=corr_mat, 
        x=hm_cols, 
        y=hm_cols,
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color="#ff8800")),
            tickfont=dict(color="#ff8800")
        )
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
        for mo in range(1, int(ten_now)+1):
            int_part = bal_run * r_now
            pri_part = emi_now - int_part
            bal_run -= pri_part
            emi_rows.append({
                "Month":         mo,
                "EMI (₹)":       round(emi_now, 2),
                "Principal (₹)": round(pri_part, 2),
                "Interest (₹)":  round(int_part, 2),
                "Balance (₹)":   round(max(bal_run, 0), 2)
            })
        sched_df = pd.DataFrame(emi_rows)
        st.dataframe(sched_df, use_container_width=True, height=320)
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
    elif menu == "📄 Loan Details (Real Bank View)":

        # ================= 🔮 CYBER STYLE (LOCAL ONLY) =================
        st.markdown("""
        <style>
        .cyber-header {
            background: linear-gradient(135deg,#050d1a,#0a1f44);
            padding:25px;
            border-radius:20px;
            color:#00ffe0;
            box-shadow:0 0 20px rgba(0,255,224,0.3);
        }

        .cyber-metric {
            background: rgba(0,255,224,0.05);
            border:1px solid rgba(0,255,224,0.2);
            border-radius:15px;
            padding:10px;
            text-align:center;
            box-shadow:0 0 10px #00ffe0;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<h1 style='color:#00ffe0'>🏦 Loan Dashboard — Cyber Banking View</h1>", unsafe_allow_html=True)

        # ================= USER DATA =================
        user_loans = loans_df[loans_df["Username"] == st.session_state.username]

        if user_loans.empty:
            st.warning("No loan applications found.")
            st.stop()

        latest = user_loans.iloc[-1]

        # ================= HEADER =================
        st.markdown(f"""
        <div class="cyber-header">
        <h2>👤 {latest['Name']}</h2>
        <p>Loan Type: {latest['Loan Type']} | Status: <b>{latest['Loan Status']}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # ================= MAIN METRICS =================
        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(f"<div class='cyber-metric'>💰 Loan<br>₹ {latest['Loan Amount']:,}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='cyber-metric'>📆 EMI<br>₹ {latest['EMI']}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='cyber-metric'>📊 Score<br>{latest['Credit Score']}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='cyber-metric'>🎯 Approval<br>{latest['Approval Probability']}%</div>", unsafe_allow_html=True)

        # ================= AI DECISION =================
        st.markdown("## 🤖 AI Decision")
        st.info(f"Risk Level: {latest['Risk']}")
        st.info(f"Fraud Check: {'⚠️ Risk' if latest['Fraud'] else 'Safe ✅'}")

        # ================= TIMELINE =================
        st.markdown("## 🧬 Loan Timeline")

        progress = int(latest["Approval Probability"])
        st.progress(progress)

        steps = ["Application", "Verification", "Credit Check", "Risk Analysis", "Approval"]

        for i, step in enumerate(steps):
            if i < progress // 20:
                st.success(f"✔ {step}")
            else:
                st.info(f"⏳ {step}")

        # ================= EMI SCHEDULE =================
        st.markdown("## 💰 EMI Schedule")

        loan_amt = float(latest["Loan Amount"])
        emi = float(latest["EMI"])
        rate = 0.1 / 12
        months = 12

        balance = loan_amt
        schedule = []

        for m in range(1, months+1):
            interest = balance * rate
            principal = emi - interest
            balance -= principal
            schedule.append([m, round(principal,2), round(interest,2), round(balance,2)])

        df_schedule = pd.DataFrame(schedule, columns=["Month","Principal","Interest","Balance"])
        st.dataframe(df_schedule, use_container_width=True)

        # ================= CYBER THEME FUNCTION =================
        def cyber_theme(fig):
            fig.update_layout(
                paper_bgcolor="#050d1a",
                plot_bgcolor="#050d1a",
                font=dict(color="#00ffe0"),
            )
            return fig

        # ================= CREDIT SCORE GAUGE =================
        st.markdown("## 📊 Credit Score")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest["Credit Score"],
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

        # ================= CHARTS =================
        st.markdown("## 📈 Financial Analytics (Advanced)")

        # ================= CYBER THEME (STRONG VISIBILITY) =================
        def cyber_theme(fig):
            fig.update_layout(
                paper_bgcolor="#050d1a",
                plot_bgcolor="#050d1a",
                font=dict(color="#00ffe0", size=12),

                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(0,255,224,0.2)",
                    zeroline=False,
                    color="#00ffe0"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(0,255,224,0.2)",
                    zeroline=False,
                    color="#00ffe0"
                ),

                legend=dict(
                    font=dict(color="#00ffe0")
                )
            )
            return fig


        # ================= 1. EMI BREAKDOWN =================
        st.markdown("### 💰 EMI Breakdown (Principal vs Interest)")

        fig_bar = go.Figure()

        fig_bar.add_bar(
            name="Principal",
            x=df_schedule["Month"],
            y=df_schedule["Principal"],
            marker=dict(color="#00ffe0")
        )

        fig_bar.add_bar(
            name="Interest",
            x=df_schedule["Month"],
            y=df_schedule["Interest"],
            marker=dict(color="#ff007f")
        )

        fig_bar.update_layout(
            barmode='stack',
            title="Monthly EMI Split"
        )

        st.plotly_chart(cyber_theme(fig_bar), use_container_width=True)


        # ================= 2. LOAN BALANCE TREND =================
        st.markdown("### 📉 Loan Balance Trend")

        fig_line = go.Figure()

        fig_line.add_trace(go.Scatter(
            x=df_schedule["Month"],
            y=df_schedule["Balance"],
            mode='lines+markers',
            name="Remaining Balance",
            line=dict(color="#00ffe0", width=3),
            marker=dict(size=6)
        ))

        fig_line.update_layout(title="Loan Balance Over Time")

        st.plotly_chart(cyber_theme(fig_line), use_container_width=True)


        # ================= 3. RISK COMPONENT ANALYSIS =================
        st.markdown("### ⚠️ Risk Component Analysis")

        risk_values = [
            latest["Credit Score"]/900 * 100,
            latest["Applicant Income"]/200000 * 100,
            latest["Approval Probability"]
        ]

        risk_labels = ["Credit Strength", "Income Strength", "Approval"]

        fig_risk = go.Figure()

        fig_risk.add_trace(go.Bar(
            x=risk_labels,
            y=risk_values,
            marker=dict(color=["#00ffe0", "#ffaa00", "#ff007f"]),
            text=[f"{v:.1f}%" for v in risk_values],
            textposition='outside'
        ))

        fig_risk.update_layout(title="Risk Component Scores")

        st.plotly_chart(cyber_theme(fig_risk), use_container_width=True)


        # ================= 4. INCOME vs OBLIGATION =================
        st.markdown("### 💼 Income vs Loan Burden")

        fig_compare = go.Figure()

        fig_compare.add_bar(
            x=["Income", "EMI"],
            y=[latest["Applicant Income"], latest["EMI"]],
            marker=dict(color=["#00ffe0", "#ff007f"])
        )

        fig_compare.update_layout(title="Income vs EMI Comparison")

        st.plotly_chart(cyber_theme(fig_compare), use_container_width=True)

        # ================= LOAN SIMULATOR =================
        st.markdown("## 🧠 Loan Simulator")

        sim_income = st.slider("Adjust Income", 10000, 200000, int(latest["Applicant Income"]))
        sim_score = st.slider("Adjust Credit Score", 300, 900, int(latest["Credit Score"]))

        sim_prob = min(100,
            (sim_score - 300)/600*50 +
            (sim_income/200000)*50
        )

        st.progress(int(sim_prob))
        st.write(f"⚡ New Approval Probability: {sim_prob:.1f}%")

        # ================= PAYMENT SIMULATION =================
        st.markdown("## 💳 Payment Simulation")

        pay = st.number_input("Enter Payment Amount", 0, int(latest["Loan Amount"]), 0)
        remaining = latest["Loan Amount"] - pay

        st.markdown(f"<div class='cyber-metric'>Remaining Balance: ₹ {remaining}</div>", unsafe_allow_html=True)

        # ================= DOWNLOAD =================
        st.markdown("## 📄 Loan Report")

        report = f"""
        Loan Report
        Name: {latest['Name']}
        Loan Amount: {latest['Loan Amount']}
        EMI: {latest['EMI']}
        Risk: {latest['Risk']}
        Approval: {latest['Approval Probability']}%
        """

        st.download_button("⬇ Download Report", report, file_name="loan_report.txt")

        # ================= FINAL DECISION =================
        st.markdown("---")

        if latest["Approval Probability"] > 70:
            st.success("✅ LOAN APPROVED")
        elif latest["Approval Probability"] > 40:
            st.warning("⚠️ UNDER REVIEW")
        else:
            st.error("❌ LOAN REJECTED")

# ===================== ADMIN SECTION =====================
if st.session_state.login and st.session_state.role == "admin":
    st.title("🛠 Admin Dashboard")

    loans_df = pd.read_csv(LOANS_FILE)
    for c in ALL_COLS:
        if c not in loans_df.columns:
            loans_df[c] = None

    st.dataframe(loans_df, use_container_width=True)
    st.subheader("📄 User Documents")

    show = [c for c in ["Name","Aadhaar","PAN","CreditFile","BankFile",
                         "Loan Status","Lifecycle Stage","Approval Probability"]
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
            loans_df.at[i, "Loan Status"]     = "Disbursed"
            loans_df.at[i, "Lifecycle Stage"] = "Loan Disbursement"
            loans_df.at[i, "Last Updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")
            loans_df.to_csv(LOANS_FILE, index=False)
            st.success(f"Application #{i} disbursed 💰")
            st.rerun()
