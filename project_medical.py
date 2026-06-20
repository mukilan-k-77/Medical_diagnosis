import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Medical Diagnosis Prediction",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #1a2744);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card h3 { color: #94a3b8; font-size: 13px; margin: 0 0 8px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: #38bdf8; font-size: 32px; margin: 0; font-weight: 700; }

    .algo-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .algo-card h4 { color: #38bdf8; margin: 0 0 4px; }
    .algo-card p  { color: #94a3b8; font-size: 13px; margin: 0; }

    .result-high {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #fca5a5;
        font-size: 20px;
        font-weight: 600;
    }
    .result-low {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #86efac;
        font-size: 20px;
        font-weight: 600;
    }

    .page-title    { font-size: 28px; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
    .page-subtitle { color: #64748b; font-size: 14px; margin-bottom: 28px; }

    hr { border-color: #334155; }

    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }

    .acc-badge {
        display: inline-block;
        background: #0c4a6e;
        color: #38bdf8;
        border: 1px solid #0284c7;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 13px;
        font-weight: 600;
    }

    .stTextInput label, .stNumberInput label,
    .stSelectbox label { color: #cbd5e1 !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================

for key, val in {
    "logged_in": False,
    "trained_models": {},
    "model_accuracies": {},
    "feature_cols": [],
    "label_encoders": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# ALGORITHMS
# ==========================================

ALGORITHMS = {
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "icon": "🌲",
        "desc": "Ensemble of decision trees; robust and accurate",
        "file": "med_model_rf.pkl",
    },
    "SVM": {
        "model": SVC(kernel="rbf", probability=True, random_state=42),
        "icon": "📐",
        "desc": "Finds optimal hyperplane with kernel trick",
        "file": "med_model_svm.pkl",
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(max_depth=10, random_state=42),
        "icon": "🌿",
        "desc": "Interpretable tree-based splits",
        "file": "med_model_dt.pkl",
    },
    "XGBoost": {
        "model": XGBClassifier(n_estimators=100, use_label_encoder=False,
                               eval_metric="logloss", random_state=42),
        "icon": "⚡",
        "desc": "Gradient boosting — often top performer",
        "file": "med_model_xgb.pkl",
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "icon": "📊",
        "desc": "Probabilistic; fast and lightweight",
        "file": "med_model_nb.pkl",
    },
}

# ==========================================
# LOGIN PAGE
# ==========================================

def login_page():
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("""
        <div style='text-align:center; padding-top:60px;'>
            <div style='font-size:52px;'>🏥</div>
            <h1 style='color:#f1f5f9; margin:8px 0 4px;'>Medical Diagnosis</h1>
            <p style='color:#64748b; margin-bottom:36px;'>Prediction System — Admin Portal</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='background:#1e293b;border:1px solid #334155;border-radius:16px;padding:36px;'>", unsafe_allow_html=True)
        st.markdown("**Username**")
        username = st.text_input("Username", label_visibility="collapsed", placeholder="Enter username")
        st.markdown("**Password**")
        password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter password")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔐  Login", use_container_width=True):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌  Invalid credentials. Try admin / admin123")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#475569;margin-top:24px;font-size:12px;'>Default: admin / admin123</p>", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAV
# ==========================================

def sidebar_nav():
    with st.sidebar:
        st.markdown("### 🏥 Medical Diagnosis")
        st.markdown("---")
        for icon, label in [("📊","Dashboard"),("🤖","Train Model"),("🩺","User Input")]:
            if st.button(f"{icon}  {label}", key=f"nav_{label}"):
                st.session_state.page = label
                st.rerun()
        st.markdown("---")
        if st.button("🚪  Logout"):
            st.session_state.logged_in = False
            st.session_state.trained_models = {}
            st.session_state.model_accuracies = {}
            st.rerun()

# ==========================================
# DASHBOARD
# ==========================================

def dashboard_page():
    st.markdown('<div class="page-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Overview of the medical dataset</div>', unsafe_allow_html=True)

    try:
        df = pd.read_csv("medical.csv")

        diabetic     = int(df["diabetes"].sum()) if "diabetes" in df.columns else 0
        non_diabetic = len(df) - diabetic
        rate         = round(diabetic / len(df) * 100, 1) if len(df) else 0

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in zip(
            [c1, c2, c3, c4],
            ["Total Records", "Diabetic", "Non-Diabetic", "Diabetic Rate"],
            [len(df), diabetic, non_diabetic, f"{rate}%"],
            ["#38bdf8", "#f87171", "#4ade80", "#fb923c"]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <h2 style='color:{color};'>{val}</h2>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("#### Class Distribution")
            fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1e293b")
            ax.set_facecolor("#1e293b")
            counts = df["diabetes"].value_counts()
            bars = ax.bar(["Non-Diabetic (0)", "Diabetic (1)"], counts.values,
                          color=["#22c55e", "#ef4444"], width=0.5, edgecolor="none")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        str(int(bar.get_height())), ha="center", color="#e2e8f0", fontsize=11)
            ax.tick_params(colors="#94a3b8")
            ax.spines[["top","right","left","bottom"]].set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_title("Diabetic vs Non-Diabetic", color="#e2e8f0", pad=10)
            plt.tight_layout()
            st.pyplot(fig)

        with ch2:
            if "age" in df.columns:
                st.markdown("#### Age Distribution by Diagnosis")
                fig2, ax2 = plt.subplots(figsize=(5, 3), facecolor="#1e293b")
                ax2.set_facecolor("#1e293b")
                for risk, color, label in [(0,"#22c55e","Non-Diabetic"), (1,"#ef4444","Diabetic")]:
                    subset = df[df["diabetes"] == risk]["age"]
                    ax2.hist(subset, bins=25, alpha=0.6, color=color, label=label, edgecolor="none")
                ax2.tick_params(colors="#94a3b8")
                ax2.spines[["top","right"]].set_visible(False)
                ax2.spines[["left","bottom"]].set_color("#334155")
                ax2.legend(facecolor="#1e293b", labelcolor="#e2e8f0")
                ax2.set_title("Age Distribution by Diagnosis", color="#e2e8f0", pad=10)
                plt.tight_layout()
                st.pyplot(fig2)

        st.markdown("---")
        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown("#### Dataset Shape")
            st.write(df.shape)
        with col_i2:
            st.markdown("#### Missing Values")
            st.write(df.isnull().sum())

    except FileNotFoundError:
        st.warning("⚠️  `medical.csv` not found. Place the dataset in the same directory.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# ==========================================
# TRAIN MODEL
# ==========================================

def train_model_page():
    st.markdown('<div class="page-title">🤖 Train Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Select algorithms and train on the medical dataset</div>', unsafe_allow_html=True)

    st.markdown("#### Choose Algorithms")
    algo_cols = st.columns(5)
    selected = []
    for i, (name, info) in enumerate(ALGORITHMS.items()):
        with algo_cols[i]:
            if st.checkbox(f"{info['icon']} {name}", value=True, key=f"chk_{name}"):
                selected.append(name)
            st.markdown(f"<p style='color:#64748b;font-size:11px;margin-top:-8px;'>{info['desc']}</p>", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀  Train Selected Models", use_container_width=False):
        if not selected:
            st.warning("Select at least one algorithm.")
            return

        try:
            df = pd.read_csv("medical.csv")
        except FileNotFoundError:
            st.error("Dataset `medical.csv` not found.")
            return

        # Encode categoricals
        le_dict = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
        st.session_state.label_encoders = le_dict

        X = df.drop("diabetes", axis=1)
        y = df["diabetes"]
        st.session_state.feature_cols = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}
        progress = st.progress(0, text="Training…")

        for idx, name in enumerate(selected):
            info = ALGORITHMS[name]
            with st.spinner(f"Training {name}…"):
                model = info["model"]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc    = accuracy_score(y_test, y_pred)

                with open(info["file"], "wb") as f:
                    pickle.dump(model, f)

                st.session_state.trained_models[name] = model
                st.session_state.model_accuracies[name] = acc
                results[name] = {"acc": acc, "y_test": y_test, "y_pred": y_pred, "info": info}

            progress.progress((idx + 1) / len(selected), text=f"Done: {name}")

        progress.empty()
        st.success(f"✅  {len(selected)} model(s) trained successfully!")

        # Accuracy table
        st.markdown("#### Accuracy Comparison")
        res_df = pd.DataFrame([
            {"Algorithm": n, "Accuracy": f"{v['acc']*100:.2f}%", "_score": v['acc']}
            for n, v in results.items()
        ]).sort_values("_score", ascending=False).drop(columns="_score")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        names = list(results.keys())
        accs  = [results[n]["acc"] * 100 for n in names]
        colors = ["#38bdf8" if a == max(accs) else "#475569" for a in accs]
        bars = ax.barh(names, accs, color=colors, edgecolor="none")
        for bar, acc in zip(bars, accs):
            ax.text(acc - 1, bar.get_y() + bar.get_height()/2,
                    f"{acc:.2f}%", va="center", ha="right", color="white", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 110)
        ax.tick_params(colors="#94a3b8")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_title("Model Accuracy (%)", color="#e2e8f0", pad=10)
        plt.tight_layout()
        st.pyplot(fig)

        # Confusion matrices
        st.markdown("---")
        st.markdown("#### Confusion Matrices")
        cm_cols = st.columns(len(results))
        for i, (name, data) in enumerate(results.items()):
            with cm_cols[i]:
                cm = confusion_matrix(data["y_test"], data["y_pred"])
                fig_cm, ax_cm = plt.subplots(figsize=(3, 2.5), facecolor="#1e293b")
                ax_cm.set_facecolor("#1e293b")
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False,
                            xticklabels=["No","Yes"], yticklabels=["No","Yes"])
                ax_cm.set_title(f"{data['info']['icon']} {name}", color="#e2e8f0", fontsize=10)
                ax_cm.tick_params(colors="#94a3b8", labelsize=8)
                plt.tight_layout()
                st.pyplot(fig_cm)

    # Previously trained summary
    if st.session_state.model_accuracies:
        st.markdown("---")
        st.markdown("#### Previously Trained Models")
        for name, acc in st.session_state.model_accuracies.items():
            info = ALGORITHMS[name]
            st.markdown(f"""
            <div class="algo-card">
                <h4>{info['icon']} {name}</h4>
                <p>Accuracy: <span class="acc-badge">{acc*100:.2f}%</span></p>
            </div>""", unsafe_allow_html=True)

# ==========================================
# USER INPUT / PREDICT
# ==========================================

def prediction_page():
    st.markdown('<div class="page-title">🩺 Diabetes Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter patient details and predict diabetes risk</div>', unsafe_allow_html=True)

    available = list(st.session_state.trained_models.keys()) or list(ALGORITHMS.keys())
    chosen = st.selectbox("🤖  Select Model for Prediction", available)

    st.markdown("---")
    st.markdown("#### Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age     = st.number_input("🎂 Age",              min_value=1,   max_value=120, value=30)
        bmi     = st.number_input("⚖️ BMI",              min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        hba1c   = st.number_input("🩸 HbA1c Level",     min_value=3.0,  max_value=15.0, value=5.5, step=0.1)
        glucose = st.number_input("💉 Blood Glucose Level", min_value=50, max_value=400, value=100)

    with col2:
        gender      = st.selectbox("👤 Gender",             ["Male", "Female", "Other"])
        hypertension = st.selectbox("💊 Hypertension",      ["No", "Yes"])
        heart_disease = st.selectbox("❤️ Heart Disease",    ["No", "Yes"])
        smoking      = st.selectbox("🚬 Smoking History",   ["never", "No Info", "current", "former", "ever", "not current"])

    st.markdown("---")

    if st.button("🔮  Predict Risk", use_container_width=False):
        # Load model
        if chosen in st.session_state.trained_models:
            model = st.session_state.trained_models[chosen]
        else:
            try:
                with open(ALGORITHMS[chosen]["file"], "rb") as f:
                    model = pickle.load(f)
            except FileNotFoundError:
                st.error("⚠️  Model not trained yet. Go to **Train Model** page first.")
                return

        # Build sample using feature cols from training
        le_dict = st.session_state.label_encoders

        def encode_val(col, val):
            if col in le_dict:
                try:
                    return int(le_dict[col].transform([val])[0])
                except:
                    return 0
            return val

        raw = {
            "gender":           encode_val("gender",          gender),
            "age":              age,
            "hypertension":     1 if hypertension  == "Yes" else 0,
            "heart_disease":    1 if heart_disease == "Yes" else 0,
            "smoking_history":  encode_val("smoking_history", smoking),
            "bmi":              bmi,
            "HbA1c_level":      hba1c,
            "blood_glucose_level": glucose,
        }

        sample = pd.DataFrame([raw])

        # Align to training columns
        if st.session_state.feature_cols:
            for col in st.session_state.feature_cols:
                if col not in sample.columns:
                    sample[col] = 0
            sample = sample[st.session_state.feature_cols]

        pred = model.predict(sample)
        prob = model.predict_proba(sample)[0] if hasattr(model, "predict_proba") else None

        st.markdown("---")
        st.markdown("#### Prediction Result")

        r1, r2 = st.columns([2, 1])
        with r1:
            if pred[0] == 1:
                st.markdown("""
                <div class="result-high">
                    ⚠️ HIGH RISK OF DIABETES<br>
                    <span style='font-size:14px;font-weight:400;'>This patient shows signs of high diabetes risk.</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-low">
                    ✅ NO DIABETES DETECTED<br>
                    <span style='font-size:14px;font-weight:400;'>This patient appears to be at low risk.</span>
                </div>""", unsafe_allow_html=True)

        with r2:
            info = ALGORITHMS[chosen]
            acc  = st.session_state.model_accuracies.get(chosen, None)
            st.markdown(f"""
            <div class="algo-card" style='margin-top:0;'>
                <h4>{info['icon']} {chosen}</h4>
                {"<p>Model Accuracy: <span class='acc-badge'>" + f"{acc*100:.2f}%" + "</span></p>" if acc else ""}
            </div>""", unsafe_allow_html=True)

            if prob is not None:
                st.markdown("**Confidence**")
                fig_p, ax_p = plt.subplots(figsize=(3, 1.8), facecolor="#1e293b")
                ax_p.set_facecolor("#1e293b")
                ax_p.barh(["No Diabetes","Diabetic"], [prob[0]*100, prob[1]*100],
                          color=["#22c55e","#ef4444"], edgecolor="none")
                for i, v in enumerate([prob[0]*100, prob[1]*100]):
                    ax_p.text(v+1, i, f"{v:.1f}%", va="center", color="#e2e8f0", fontsize=9)
                ax_p.set_xlim(0, 115)
                ax_p.tick_params(colors="#94a3b8", labelsize=8)
                ax_p.spines[["top","right","left","bottom"]].set_visible(False)
                ax_p.xaxis.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_p)

        # Summary
        st.markdown("---")
        with st.expander("📋 View Patient Summary"):
            summary = {
                "Gender": gender, "Age": age, "BMI": bmi,
                "HbA1c Level": hba1c, "Blood Glucose": glucose,
                "Hypertension": hypertension, "Heart Disease": heart_disease,
                "Smoking History": smoking,
            }
            st.dataframe(pd.DataFrame(summary.items(), columns=["Field","Value"]),
                         use_container_width=True, hide_index=True)

# ==========================================
# PAGE ROUTING
# ==========================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if not st.session_state.logged_in:
    login_page()
else:
    sidebar_nav()

    if st.session_state.page == "Dashboard":
        dashboard_page()
    elif st.session_state.page == "Train Model":
        train_model_page()
    elif st.session_state.page == "User Input":
        prediction_page()
