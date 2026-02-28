"""
app/app.py – Streamlit app for interactive Titanic survival prediction.

Run:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container { padding: 2rem 3rem; }

    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 { font-size: 2.8rem; font-weight: 700; margin: 0; }
    .hero p  { font-size: 1.1rem; opacity: 0.8; margin-top: 0.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        color: white;
    }
    .metric-label { font-size: 0.8rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 2rem; font-weight: 700; }

    .result-survived {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(56, 239, 125, 0.3);
    }
    .result-not-survived {
        background: linear-gradient(135deg, #c94b4b, #4b134f);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(201, 75, 75, 0.3);
    }
    .stSlider > div > div { background: rgba(255,255,255,0.05); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🚢 Titanic Survival Predictor</h1>
        <p>Enter passenger details to predict survival probability using ML</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load artefacts ────────────────────────────────────────────────────────────
MODEL_PATH  = ROOT / "models" / "best_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_artifacts()

if model is None:
    st.error(
        "⚠️ Model not found. Please run `python main.py` first to train and save the model.",
        icon="🔴",
    )
    st.stop()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧑 Passenger Details")
    st.markdown("---")

    pclass   = st.selectbox("Passenger Class", [1, 2, 3], index=2,
                             help="1 = First, 2 = Second, 3 = Third")
    sex      = st.radio("Sex", ["Male", "Female"])
    age      = st.slider("Age", 0, 80, 28)
    sibsp    = st.slider("Siblings / Spouses Aboard (SibSp)", 0, 8, 0)
    parch    = st.slider("Parents / Children Aboard (Parch)", 0, 6, 0)
    fare     = st.slider("Fare Paid (£)", 0.0, 520.0, 32.0, step=0.5)
    embarked = st.selectbox("Port of Embarkation",
                             ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
    title_opt = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Survival", use_container_width=True)

# ── Derived features ──────────────────────────────────────────────────────────
sex_enc      = 0 if sex == "Male" else 1
embarked_enc = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}[embarked]
title_enc    = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}[title_opt]
family_size  = sibsp + parch + 1
is_alone     = int(family_size == 1)
fare_per_p   = fare / family_size

FEATURE_COLS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    "Title", "FamilySize", "IsAlone", "FarePerPerson",
]

input_data = pd.DataFrame([[
    pclass, sex_enc, age, sibsp, parch, fare,
    embarked_enc, title_enc, family_size, is_alone, fare_per_p,
]], columns=FEATURE_COLS)

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Passenger Class</div>
            <div class="metric-value">{'⭐' * pclass}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Family Size</div>
            <div class="metric-value">{family_size}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Fare per Person</div>
            <div class="metric-value">£{fare_per_p:.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    # determine which columns the scaler expects
    try:
        n_features = scaler.n_features_in_
        available  = input_data.columns.tolist()[:n_features]
        scaled     = scaler.transform(input_data[available])
        proba      = model.predict_proba(scaled)[0]
        pred       = model.predict(scaled)[0]
        survival_pct = proba[1] * 100

        col_a, col_b = st.columns([2, 1])

        with col_a:
            if pred == 1:
                st.markdown(
                    f"""<div class="result-survived">
                        ✅ <strong>SURVIVED</strong><br>
                        <span style="font-size:0.9rem; opacity:0.9;">
                        This passenger is predicted to have survived the Titanic disaster.
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="result-not-survived">
                        ❌ <strong>DID NOT SURVIVE</strong><br>
                        <span style="font-size:0.9rem; opacity:0.9;">
                        This passenger is predicted to not have survived.
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        with col_b:
            st.metric("Survival Probability", f"{survival_pct:.1f}%")
            st.metric("Not-Survival Probability", f"{proba[0]*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bar
        st.write("**Survival Probability Breakdown**")
        prob_df = pd.DataFrame({
            "Outcome": ["Survived", "Did Not Survive"],
            "Probability": [proba[1], proba[0]],
        }).set_index("Outcome")
        st.bar_chart(prob_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("👈 Fill in the passenger details on the left and click **Predict Survival**.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Scikit-Learn & Streamlit · "
    "Titanic Dataset from OpenML</small></center>",
    unsafe_allow_html=True,
)
