import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="RetentionAI | Employee Turnover Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ADVANCED CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* GLOBAL VARIABLES */
    :root {
        --primary: #1E3A5F;
        --secondary: #2E7D9E;
        --accent: #ff0080;
        --success: #10B981;
        --danger: #EF4444;
        --warning: #F59E0B;
        --bg-light: #F8FAFC;
        --card-bg: rgba(255, 255, 255, 0.9);
        --glass-border: rgba(255, 255, 255, 0.5);
    }

    /* GENERAL SETTINGS */
    .stApp {
        background: linear-gradient(135deg, #F0F4F8 0%, #E2E8F0 100%);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: var(--primary);
    }

    /* CARD DESIGN WITH GLASSMORPHISM */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border-color: rgba(255, 255, 255, 0.8);
    }

    /* HEADERS */
    .main-header {
        background: linear-gradient(90deg, #1E3A5F 0%, #2E7D9E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /* INPUT STYLING */
    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        padding: 0.5rem;
        transition: all 0.3s;
    }
    
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--secondary);
        box-shadow: 0 0 0 3px rgba(46, 125, 158, 0.1);
    }

    /* CUSTOM PREDICT BUTTON (Rainbow & Glow) */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #ff0080, #ff8c00, #40e0d0, #ff0080);
        background-size: 400% 400%;
        color: white !important;
        font-size: 1.3rem;
        font-weight: 800 !important;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none !important;
        cursor: pointer;
        box-shadow: 0 10px 20px rgba(255, 0, 128, 0.3);
        transition: all 0.4s ease;
        animation: gradientShift 4s ease infinite;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 30px rgba(255, 0, 128, 0.5);
    }
    
    .stButton>button:active {
        transform: translateY(1px);
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* RESULT BOXES */
    .result-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 1rem;
        text-align: center;
    }
    
    .status-badge {
        font-size: 1.5rem;
        font-weight: 800;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    .result-leave {
        background: rgba(254, 226, 226, 0.8);
        border: 2px solid var(--danger);
        color: #991B1B;
        animation: pulse-red 2s infinite;
    }
    
    .result-stay {
        background: rgba(209, 250, 229, 0.8);
        border: 2px solid var(--success);
        color: #065F46;
        animation: pulse-green 2s infinite;
    }

    /* CUSTOM PROGRESS BARS */
    .prob-container {
        width: 100%;
        background-color: #E2E8F0;
        border-radius: 10px;
        height: 12px;
        margin-top: 8px;
        overflow: hidden;
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #fff;
        border-right: 1px solid #E2E8F0;
    }
    
    /* FEATURE ICONS */
    .feature-icon {
        font-size: 1.2rem;
        margin-right: 8px;
        display: inline-block;
    }
    
    /* METRIC CARDS */
    .metric-box {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid var(--secondary);
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
HF_REPO_ID = "Zlib2/RFC"
MODEL_FILENAME = "final_random_forest_model.joblib"

# Define the 5 selected features in exact model order
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
    """Load model from Hugging Face with caching and error handling"""
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================================================================
# STATE MANAGEMENT (Sync Sliders & Inputs)
# ============================================================================
def sync_sat_slider(): st.session_state.sat_val = st.session_state.sat_slider
def sync_sat_input(): st.session_state.sat_slider = st.session_state.sat_val
def sync_eval_slider(): st.session_state.eval_val = st.session_state.eval_slider
def sync_eval_input(): st.session_state.eval_slider = st.session_state.eval_val

if 'sat_val' not in st.session_state: st.session_state.sat_val = 0.50
if 'sat_slider' not in st.session_state: st.session_state.sat_slider = 0.50
if 'eval_val' not in st.session_state: st.session_state.eval_val = 0.70
if 'eval_slider' not in st.session_state: st.session_state.eval_slider = 0.70

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================
def main():
    # Load Model
    model = load_model()

    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 Model Intelligence")
        st.markdown("""
        <div class="metric-box">
            <b>Repository:</b><br>Hugging Face (Zlib2/RFC)<br><br>
            <b>Algorithm:</b><br>Random Forest Classifier<br><br>
            <b>Accuracy:</b><br>High Precision (Balanced)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Features Analyzed")
        features_html = ""
        icons = ["😊", "⏱️", "⏰", "📁", "📈"]
        for i, feat in enumerate(BEST_FEATURES):
            name = feat.replace('_', ' ').title()
            features_html += f"<div style='margin-bottom:8px;'><span class='feature-icon'>{icons[i]}</span>{name}</div>"
        st.markdown(features_html, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("v1.0.0 | Enterprise Edition")

    # Main Header
    st.markdown('<h1 class="main-header">RetentionAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Employee Turnover Prediction System</p>', unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Model could not be loaded. Please check your internet connection or Hugging Face status.")
        return

    # Input Section Wrapper
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Employee Profile")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROW 1: Satisfaction & Evaluation ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### 😊 Satisfaction Level")
        sub_c1, sub_c2 = st.columns([3, 1])
        with sub_c1:
            st.slider("Sat Slider", 0.0, 1.0, key="sat_slider", on_change=sync_sat_slider, label_visibility="collapsed")
        with sub_c2:
            st.number_input("Sat Input", 0.0, 1.0, step=0.01, key="sat_val", on_change=sync_sat_input, label_visibility="collapsed")
        st.caption("0.0 = Very Dissatisfied | 1.0 = Very Satisfied")

    with c2:
        st.markdown("##### 📈 Last Evaluation")
        sub_c1, sub_c2 = st.columns([3, 1])
        with sub_c1:
            st.slider("Eval Slider", 0.0, 1.0, key="eval_slider", on_change=sync_eval_slider, label_visibility="collapsed")
        with sub_c2:
            st.number_input("Eval Input", 0.0, 1.0, step=0.01, key="eval_val", on_change=sync_eval_input, label_visibility="collapsed")
        st.caption("0.0 = Poor Performance | 1.0 = Excellent Performance")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROW 2: Quantitative Metrics ---
    c3, c4, c5 = st.columns(3)
    
    with c3:
        st.markdown("##### ⏱️ Years at Company")
        time_spend = st.number_input("Years", min_value=1, max_value=20, value=3, step=1, label_visibility="collapsed")
    
    with c4:
        st.markdown("##### 📁 Number of Projects")
        num_projects = st.number_input("Projects", min_value=1, max_value=10, value=4, step=1, label_visibility="collapsed")
        
    with c5:
        st.markdown("##### ⏰ Monthly Hours")
        avg_hours = st.number_input("Hours", min_value=50, max_value=350, value=180, step=5, label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True) # End Input Card

    # --- ACTION AREA ---
    col_spacer_l, col_btn, col_spacer_r = st.columns([1, 2, 1])
    with col_btn:
        predict_btn = st.button("✨ ANALYZE RETENTION RISK ✨")

    # --- PREDICTION LOGIC ---
    if predict_btn:
        # Show a quick spinner for realism/ux
        with st.spinner("Processing algorithmic analysis..."):
            time.sleep(0.6) # Artificial delay for UX
            
            # Prepare Data
            input_data = pd.DataFrame([{
                'satisfaction_level': st.session_state.sat_val,
                'time_spend_company': time_spend,
                'average_monthly_hours': avg_hours,
                'number_project': num_projects,
                'last_evaluation': st.session_state.eval_val
            }])[BEST_FEATURES]

            # Predict
            pred_class = model.predict(input_data)[0]
            pred_proba = model.predict_proba(input_data)[0]
            
            prob_stay = pred_proba[0] * 100
            prob_leave = pred_proba[1] * 100

        # --- RESULTS DISPLAY ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Determine styling based on result
        if pred_class == 1:
            verdict = "HIGH TURNOVER RISK"
            sub_verdict = "The model predicts this employee is likely to **LEAVE**."
            style_class = "result-leave"
            icon = "⚠️"
        else:
            verdict = "LOW TURNOVER RISK"
            sub_verdict = "The model predicts this employee is likely to **STAY**."
            style_class = "result-stay"
            icon = "✅"

        # Result Card
        st.markdown(f"""
        <div class="glass-card result-container {style_class}">
            <div class="status-badge {style_class}" style="background:white; border:none;">{icon} {verdict}</div>
            <p style="font-size: 1.2rem;">{sub_verdict}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probabilities Section
        c_res1, c_res2 = st.columns(2)
        
        with c_res1:
            st.markdown(f"""
            <div class="metric-box">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <strong>Probability of Staying</strong>
                    <strong>{prob_stay:.1f}%</strong>
                </div>
                <div class="prob-container">
                    <div class="prob-fill" style="width: {prob_stay}%; background: linear-gradient(90deg, #10B981, #34D399);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_res2:
            st.markdown(f"""
            <div class="metric-box">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <strong>Probability of Leaving</strong>
                    <strong>{prob_leave:.1f}%</strong>
                </div>
                <div class="prob-container">
                    <div class="prob-fill" style="width: {prob_leave}%; background: linear-gradient(90deg, #EF4444, #F87171);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Input Summary Expander
        with st.expander("🔍 View Analysis Parameters"):
            st.table(pd.DataFrame({
                'Metric': ['Satisfaction', 'Evaluation', 'Tenure', 'Projects', 'Avg Hours'],
                'Value': [
                    f"{st.session_state.sat_val:.2f}", 
                    f"{st.session_state.eval_val:.2f}", 
                    f"{time_spend} years", 
                    f"{num_projects}", 
                    f"{avg_hours} hrs"
                ]
            }))

if __name__ == "__main__":
    main()
