import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
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
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* ===== FONT IMPORT ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap');
    
    /* ===== GLOBAL STYLES ===== */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ===== HEADER STYLES ===== */
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        color: #1E3A5F; /* Dark Professional Blue */
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.25rem;
        color: #5A6A7D; /* Muted Slate Gray */
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /* ===== CARD & CONTAINER STYLES ===== */
    .content-card {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #EAEFF5;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    
    /* ===== FORM INPUT STYLES ===== */
    .st-emotion-cache-13mg6ur { /* Number Input */
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #DDE2E8;
        transition: border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .st-emotion-cache-13mg6ur:focus-within {
        border-color: #007BFF;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.15);
    }
    .st-emotion-cache-1kyxreq { /* Slider Track */
        height: 8px !important;
        border-radius: 4px;
    }
    .st-emotion-cache-13mg6ur p { /* Input Label */
        font-weight: 500;
        color: #334155;
    }

    /* ===== PREDICTION BUTTON STYLES ===== */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3); }
        50% { transform: scale(1.02); box-shadow: 0 6px 25px rgba(0, 123, 255, 0.5); }
        100% { transform: scale(1); box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3); }
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #007BFF, #0056b3, #00AFFF, #007BFF);
        background-size: 400% 400%;
        color: white !important;
        font-size: 1.5rem;
        font-weight: 700 !important;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none !important;
        cursor: pointer;
        animation: gradientShift 4s ease infinite, pulse 2.5s ease-in-out infinite;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 123, 255, 0.4);
        animation-play-state: paused, running;
    }
    .stButton>button:active {
        transform: translateY(1px) scale(0.99);
        box-shadow: 0 2px 10px rgba(0, 123, 255, 0.3);
    }

    /* ===== RESULT BOX STYLES ===== */
    .result-box {
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid transparent;
        transition: all 0.5s ease-in-out;
    }
    .stay-prediction {
        background-color: #E6F9EE; /* Light Green */
        border-color: #28a745;
    }
    .leave-prediction {
        background-color: #FDEEEE; /* Light Red */
        border-color: #dc3545;
    }
    .result-box h1 {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
    }
    .stay-prediction h1 { color: #28a745; }
    .leave-prediction h1 { color: #dc3545; }
    .result-box p {
        font-size: 1.25rem;
        color: #334155;
        margin-top: 0.5rem;
    }

    /* ===== PROGRESS BAR STYLES ===== */
    .progress-bar-container {
        width: 100%;
        background-color: #E9ECEF;
        border-radius: 50px;
        margin: 0.5rem 0 1rem 0;
        height: 20px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .progress-bar-green {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #218838);
        border-radius: 50px;
        transition: width 0.8s cubic-bezier(0.25, 0.1, 0.25, 1);
    }
    .progress-bar-red {
        height: 100%;
        background: linear-gradient(90deg, #dc3545, #c82333);
        border-radius: 50px;
        transition: width 0.8s cubic-bezier(0.25, 0.1, 0.25, 1);
    }
    
    /* ===== SIDEBAR STYLES ===== */
    div[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #EAEFF5;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #DDE2E8 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF;
    }
    div[data-testid="stExpander"] details summary {
        background-color: #1E3A5F !important;
        color: white !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.75rem 1rem !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 8px 8px 0 0 !important;
    }
    div[data-testid="stExpander"] details > div {
        border-radius: 0 0 8px 8px !important;
    }

    /* ===== ANIMATION UTILITIES ===== */
    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
HF_REPO_ID = "Zlib2/RFC"
MODEL_FILENAME = "final_random_forest_model.joblib"

BEST_FEATURES = [
    "satisfaction_level", "time_spend_company", "average_monthly_hours",
    "number_project", "last_evaluation"
]

# ============================================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================================
@st.cache_resource
def load_model_from_huggingface():
    """Load the trained model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=MODEL_FILENAME, repo_type="model"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ============================================================================
# CALLBACK FUNCTIONS FOR SYNCING SLIDERS AND NUMBER INPUTS
# ============================================================================
def sync_satisfaction_slider():
    st.session_state.satisfaction_level = st.session_state.sat_slider
def sync_satisfaction_input():
    st.session_state.satisfaction_level = st.session_state.sat_input
def sync_evaluation_slider():
    st.session_state.last_evaluation = st.session_state.eval_slider
def sync_evaluation_input():
    st.session_state.last_evaluation = st.session_state.eval_input

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header fade-in">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header fade-in">Predict whether an employee is likely to leave the company</p>', unsafe_allow_html=True)
    
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        return
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.header("! Model Information")
        st.markdown(f"""
        **Repository:** `{HF_REPO_ID}`<br>
        **Algorithm:** Random Forest Classifier<br>
        **Features Used:** {len(BEST_FEATURES)}<br>
        **Class Weight:** Balanced
        """)
        st.markdown("---")
        st.subheader("📋 Selected Features")
        for i, feat in enumerate(BEST_FEATURES, 1):
            feat_display = feat.replace('_', ' ').title()
            st.write(f"{i}. {feat_display}")
        st.markdown("---")
        st.subheader("🎯 Target Variable")
        st.info("""
        - **0 → Stay**: Employee likely to stay
        - **1 → Leave**: Employee likely to leave
        """)
        st.markdown("---")
        st.markdown(f"[🔗 View on Hugging Face](https://huggingface.co/{HF_REPO_ID})")
    
    # ========================================================================
    # MAIN INPUT SECTION
    # ========================================================================
    st.markdown("### 📝 Enter Employee Information")
    
    # Initialize session state
    if 'satisfaction_level' not in st.session_state: st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state: st.session_state.last_evaluation = 0.7
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.write("😊 **Satisfaction Level**")
            sat_c1, sat_c2 = st.columns([3, 1])
            with sat_c1:
                st.slider("Satisfaction", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, key="sat_slider", on_change=sync_satisfaction_slider)
            with sat_c2:
                st.number_input("Satisfaction", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, "%.2f", key="sat_input", on_change=sync_satisfaction_input)
            satisfaction_level = st.session_state.satisfaction_level
        with col2:
            st.write("📊 **Last Evaluation Score**")
            eval_c1, eval_c2 = st.columns([3, 1])
            with eval_c1:
                st.slider("Evaluation", 0.0, 1.0, st.session_state.last_evaluation, 0.01, key="eval_slider", on_change=sync_evaluation_slider)
            with eval_c2:
                st.number_input("Evaluation", 0.0, 1.0, st.session_state.last_evaluation, 0.01, "%.2f", key="eval_input", on_change=sync_evaluation_input)
            last_evaluation = st.session_state.last_evaluation

    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            time_spend_company = st.number_input("📅 Years at Company", 1, 40, 3, 1)
        with col2:
            number_project = st.number_input("📁 Number of Projects", 1, 10, 4, 1)
        with col3:
            average_monthly_hours = st.number_input("⏰ Average Monthly Hours", 80, 350, 200, 5)
    
    input_data = {
        'satisfaction_level': satisfaction_level, 'time_spend_company': time_spend_company,
        'average_monthly_hours': average_monthly_hours, 'number_project': number_project,
        'last_evaluation': last_evaluation
    }
    
    # ========================================================================
    # PREDICTION BUTTON
    # ========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_button, _ = st.columns([2, 3, 2])
    with col_button:
        predict_button = st.button("🔮 Predict Employee Turnover", use_container_width=True)

    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================
    if predict_button:
        with st.spinner("Analyzing data and generating prediction..."):
            input_df = pd.DataFrame([input_data])[BEST_FEATURES]
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            prob_stay, prob_leave = prediction_proba[0] * 100, prediction_proba[1] * 100

        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="result-box stay-prediction fade-in">
                    <h1>✅ STAY</h1>
                    <p>Employee is likely to <strong>STAY</strong> with the company.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box leave-prediction fade-in">
                    <h1>⚠️ LEAVE</h1>
                    <p>Employee is likely to <strong>LEAVE</strong> the company.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Prediction Probabilities")
            st.write(f"**Probability of Staying:** {prob_stay:.1f}%")
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-green" style="width: {prob_stay}%;"></div></div>', unsafe_allow_html=True)
            
            st.write(f"**Probability of Leaving:** {prob_leave:.1f}%")
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-red" style="width: {prob_leave}%;"></div></div>', unsafe_allow_html=True)
        
        # ====================================================================
        # INPUT SUMMARY (Collapsible Dropdown)
        # ====================================================================
        st.markdown("---")
        _, col_summary, _ = st.columns([2, 3, 2])
        with col_summary:
            with st.expander("📋 Click to View Input Summary", expanded=False):
                summary_df = pd.DataFrame({
                    'Feature': ['😊 Satisfaction Level', '📊 Last Evaluation', '📅 Years at Company', '📁 Number of Projects', '⏰ Average Monthly Hours'],
                    'Value': [f"{satisfaction_level:.2f}", f"{last_evaluation:.2f}", f"{time_spend_company} years", f"{number_project} projects", f"{average_monthly_hours} hours"]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
