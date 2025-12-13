import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING (REDUCED BUTTON MOVEMENT)
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* ================= BUTTON ================= */
    .stButton>button {
        width: 100%;
        background: linear-gradient(
            45deg,
            #ff0080, #ff8c00, #40e0d0, #ff0080
        );
        background-size: 400% 400%;
        color: white !important;
        font-size: 1.4rem;
        font-weight: bold;
        padding: 1.2rem 2.5rem;
        border-radius: 50px;
        border: none;
        cursor: pointer;
        position: relative;
        overflow: hidden;

        box-shadow:
            0 4px 15px rgba(255, 0, 128, 0.4),
            0 8px 30px rgba(255, 140, 0, 0.3);

        animation: gradientShift 3s ease infinite, pulse 2s ease-in-out infinite;

        transition:
            transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1),
            box-shadow 0.3s ease,
            background 0.4s ease;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* REDUCED pulse */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }

    /* REDUCED hover movement */
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow:
            0 8px 25px rgba(0, 245, 255, 0.4),
            0 12px 40px rgba(255, 0, 255, 0.3);
    }

    .stButton>button:active {
        transform: translateY(1px) scale(0.99);
    }

    /* Ensure text stays white */
    .stButton>button,
    .stButton>button:hover,
    .stButton>button:active {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
HF_REPO_ID = "Zlib2/RFC"
MODEL_FILENAME = "final_random_forest_model.joblib"

BEST_FEATURES = [
    "satisfaction_level",
    "time_spend_company",
    "average_monthly_hours",
    "number_project",
    "last_evaluation"
]

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )
    return joblib.load(path)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether an employee is likely to leave</p>', unsafe_allow_html=True)

    model = load_model()

    st.markdown("### 📝 Employee Information")

    satisfaction = st.slider("😊 Satisfaction Level", 0.0, 1.0, 0.5, 0.01)
    evaluation = st.slider("📊 Last Evaluation", 0.0, 1.0, 0.7, 0.01)
    years = st.number_input("📅 Years at Company", 1, 40, 3)
    projects = st.number_input("📁 Number of Projects", 1, 10, 4)
    hours = st.number_input("⏰ Average Monthly Hours", 80, 350, 200, 5)

    input_df = pd.DataFrame([{
        "satisfaction_level": satisfaction,
        "time_spend_company": years,
        "average_monthly_hours": hours,
        "number_project": projects,
        "last_evaluation": evaluation
    }])[BEST_FEATURES]

    st.markdown("---")

    if st.button("🔮 Predict Employee Turnover"):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        if pred == 0:
            st.success(f"✅ Employee likely to STAY ({proba[0]*100:.1f}%)")
        else:
            st.error(f"⚠️ Employee likely to LEAVE ({proba[1]*100:.1f}%)")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()
