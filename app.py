import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Employee Turnover Prediction | HR Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING - PROFESSIONAL & MODERN DESIGN
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    
    /* ===================== GLOBAL STYLES ===================== */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        animation: gradientAnimation 15s ease infinite;
    }
    
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* ===================== HEADER SECTION ===================== */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        animation: fadeInUp 1s ease-out 0.2s both;
    }
    
    .header-icon {
        font-size: 4rem;
        text-align: center;
        margin-bottom: 1rem;
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* ===================== SIDEBAR STYLING ===================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .metric-card:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* ===================== INPUT CARDS ===================== */
    .input-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 
                    0 8px 16px rgba(0, 0, 0, 0.05);
        margin: 1.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid #e2e8f0;
        animation: fadeIn 0.6s ease-out;
    }
    
    .input-card:hover {
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1), 
                    0 16px 32px rgba(0, 0, 0, 0.08);
        transform: translateY(-4px);
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .feature-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .feature-description {
        font-size: 0.9rem;
        color: #718096;
        margin-bottom: 1rem;
        font-style: italic;
    }
    
    /* ===================== PREDICTION BUTTON ===================== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(
            135deg, 
            #667eea 0%, 
            #764ba2 50%, 
            #f093fb 100%
        );
        background-size: 200% 200%;
        color: white !important;
        font-size: 1.5rem;
        font-weight: 700 !important;
        padding: 1.5rem 3rem;
        border-radius: 50px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 10px 25px rgba(102, 126, 234, 0.4),
            0 6px 12px rgba(118, 75, 162, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: gradientShift 4s ease infinite;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 
            0 15px 35px rgba(102, 126, 234, 0.5),
            0 8px 16px rgba(118, 75, 162, 0.4),
            0 0 40px rgba(240, 147, 251, 0.3);
        animation: gradientShift 2s ease infinite;
    }
    
    .stButton>button:active {
        transform: translateY(-2px) scale(0.98);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            120deg,
            transparent,
            rgba(255, 255, 255, 0.3),
            transparent
        );
        transition: left 0.7s ease;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button::after {
        content: '✨';
        position: absolute;
        font-size: 1.5rem;
        right: 30px;
        animation: sparkle 1.5s ease-in-out infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { opacity: 1; transform: scale(1) rotate(0deg); }
        50% { opacity: 0.6; transform: scale(1.2) rotate(180deg); }
    }
    
    /* ===================== PREDICTION RESULTS ===================== */
    .prediction-box {
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        animation: scaleIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.5) rotateX(90deg);
        }
        to {
            opacity: 1;
            transform: scale(1) rotateX(0deg);
        }
    }
    
    .stay-prediction {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 3px solid #28a745;
    }
    
    .leave-prediction {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 3px solid #dc3545;
    }
    
    .prediction-box h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-box p {
        font-size: 1.4rem;
        font-weight: 500;
    }
    
    .prediction-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
        animation: iconPulse 2s ease-in-out infinite;
    }
    
    @keyframes iconPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    /* ===================== PROBABILITY CARDS ===================== */
    .probability-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        animation: slideInRight 0.6s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .probability-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.8rem;
    }
    
    .probability-value {
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        margin-bottom: 1rem;
    }
    
    .prob-stay {
        color: #28a745;
    }
    
    .prob-leave {
        color: #dc3545;
    }
    
    /* ===================== PROGRESS BARS ===================== */
    .progress-bar-container {
        width: 100%;
        background: #e2e8f0;
        border-radius: 50px;
        margin: 1rem 0;
        height: 30px;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    
    .progress-bar-green {
        height: 100%;
        background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
        border-radius: 50px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(72, 187, 120, 0.4);
    }
    
    .progress-bar-red {
        height: 100%;
        background: linear-gradient(90deg, #fc8181 0%, #f56565 100%);
        border-radius: 50px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(245, 101, 101, 0.4);
    }
    
    .progress-bar-green::after,
    .progress-bar-red::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.3),
            transparent
        );
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .progress-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-weight: 700;
        color: #2d3748;
        font-size: 0.95rem;
        z-index: 10;
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
    }
    
    /* ===================== LOADING SPINNER ===================== */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        animation: fadeIn 0.3s ease-out;
    }
    
    .spinner {
        width: 60px;
        height: 60px;
        border: 5px solid #e2e8f0;
        border-top: 5px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        margin-top: 1.5rem;
        font-size: 1.2rem;
        color: #4a5568;
        font-weight: 500;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* ===================== EXPANDER STYLING ===================== */
    div[data-testid="stExpander"] {
        border: none !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        animation: fadeIn 0.5s ease-out;
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        cursor: pointer;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    div[data-testid="stExpander"] details summary svg {
        color: white !important;
        fill: white !important;
    }
    
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 12px 12px 0 0 !important;
    }
    
    div[data-testid="stExpander"] details > div {
        border: 2px solid #667eea !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        background: white;
        padding: 1.5rem !important;
    }
    
    /* ===================== INFO BOXES ===================== */
    .info-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        animation: fadeIn 0.5s ease-out;
        box-shadow: 0 4px 6px rgba(245, 158, 11, 0.1);
    }
    
    .info-box strong {
        color: #92400e;
    }
    
    /* ===================== METRIC CARDS IN MAIN CONTENT ===================== */
    .metric-card-main {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card-main:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transform: translateX(5px);
    }
    
    /* ===================== SLIDERS & INPUTS ===================== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* ===================== TOOLTIPS ===================== */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #2d3748;
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* ===================== RESPONSIVE DESIGN ===================== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
        }
        
        .prediction-box h1 {
            font-size: 2.5rem;
        }
        
        .probability-value {
            font-size: 2rem;
        }
        
        .stButton>button {
            font-size: 1.2rem;
            padding: 1.2rem 2rem;
        }
    }
    
    /* ===================== DIVIDER ===================== */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
    
    /* ===================== DATAFRAME STYLING ===================== */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* ===================== FEATURE BOX ===================== */
    .feature-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 2px solid #818cf8;
        animation: fadeIn 0.5s ease-out;
    }
    
    .feature-box strong {
        color: #3730a3;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
HF_REPO_ID = "Zlib2/RFC"
MODEL_FILENAME = "final_random_forest_model.joblib"

# Define the 5 selected features (in exact order)
BEST_FEATURES = [
    "satisfaction_level",
    "time_spend_company", 
    "average_monthly_hours",
    "number_project",
    "last_evaluation"
]

# ============================================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================================
@st.cache_resource
def load_model_from_huggingface():
    """Load the trained model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
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
    # Animated Header with Icon
    st.markdown('<div class="header-icon">👥</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered HR Analytics Platform</p>', unsafe_allow_html=True)
    
    # Load model silently
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown("### ℹ️ Model Information")
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>🤗 Repository</strong><br>
            <span style="font-size: 0.9rem;">{HF_REPO_ID}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <strong>🤖 Algorithm</strong><br>
            Random Forest Classifier
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>📊 Features</strong><br>
            {len(BEST_FEATURES)} Key Predictors
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <strong>⚖️ Model Balance</strong><br>
            Class-Weighted Training
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 Features Used")
        feature_icons = ["😊", "📅", "⏰", "📁", "📊"]
        for icon, feat in zip(feature_icons, BEST_FEATURES):
            feat_display = feat.replace('_', ' ').title()
            st.markdown(f"{icon} **{feat_display}**")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Prediction Classes")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <strong style="color: #4ade80;">✅ Stay (0)</strong><br>
            <span style="font-size: 0.9rem;">Employee likely to remain</span>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <strong style="color: #f87171;">⚠️ Leave (1)</strong><br>
            <span style="font-size: 0.9rem;">Employee at risk of leaving</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"[🔗 View on Hugging Face](https://huggingface.co/{HF_REPO_ID})")
    
    # ========================================================================
    # MAIN INPUT SECTION
    # ========================================================================
    st.markdown("---")
    
    st.markdown("""
    <div class="feature-box">
        <strong>📌 Instructions</strong><br>
        Please provide the employee information below to generate an AI-powered turnover prediction.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # INPUT CARDS
    # ========================================================================
    
    # ROW 1: Satisfaction & Evaluation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="input-card">
            <div class="feature-label">😊 Satisfaction Level</div>
            <div class="feature-description">Employee's self-reported job satisfaction (0 = Very Low, 1 = Very High)</div>
        </div>
        """, unsafe_allow_html=True)
        
        sat_col1, sat_col2 = st.columns([3, 1])
        with sat_col1:
            st.slider(
                "Satisfaction Slider",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.satisfaction_level,
                step=0.01,
                label_visibility="collapsed",
                key="sat_slider",
                on_change=sync_satisfaction_slider
            )
        with sat_col2:
            st.number_input(
                "Satisfaction Input",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.satisfaction_level,
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key="sat_input",
                on_change=sync_satisfaction_input
            )
        
        satisfaction_level = st.session_state.satisfaction_level
    
    with col2:
        st.markdown("""
        <div class="input-card">
            <div class="feature-label">📊 Last Evaluation Score</div>
            <div class="feature-description">Most recent performance review rating (0 = Poor, 1 = Excellent)</div>
        </div>
        """, unsafe_allow_html=True)
        
        eval_col1, eval_col2 = st.columns([3, 1])
        with eval_col1:
            st.slider(
                "Evaluation Slider",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.last_evaluation,
                step=0.01,
                label_visibility="collapsed",
                key="eval_slider",
                on_change=sync_evaluation_slider
            )
        with eval_col2:
            st.number_input(
                "Evaluation Input",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.last_evaluation,
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key="eval_input",
                on_change=sync_evaluation_input
            )
        
        last_evaluation = st.session_state.last_evaluation
    
    # ROW 2: Years & Projects
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="input-card">
            <div class="feature-label">📅 Years at Company</div>
            <div class="feature-description">Total tenure with the organization</div>
        </div>
        """, unsafe_allow_html=True)
        
        time_spend_company = st.number_input(
            "Years at Company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            label_visibility="collapsed"
        )
    
    with col4:
        st.markdown("""
        <div class="input-card">
            <div class="feature-label">📁 Number of Projects</div>
            <div class="feature-description">Active projects assigned to employee</div>
        </div>
        """, unsafe_allow_html=True)
        
        number_project = st.number_input(
            "Number of Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            label_visibility="collapsed"
        )
    
    # ROW 3: Monthly Hours
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("""
        <div class="input-card">
            <div class="feature-label">⏰ Average Monthly Hours</div>
            <div class="feature-description">Average hours worked per month</div>
        </div>
        """, unsafe_allow_html=True)
        
        average_monthly_hours = st.number_input(
            "Average Monthly Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            label_visibility="collapsed"
        )
    
    # Create input dictionary
    input_data = {
        'satisfaction_level': satisfaction_level,
        'time_spend_company': time_spend_company,
        'average_monthly_hours': average_monthly_hours,
        'number_project': number_project,
        'last_evaluation': last_evaluation
    }
    
    # ========================================================================
    # PREDICTION BUTTON
    # ========================================================================
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Turnover Risk", use_container_width=True)
    
    # ========================================================================
    # PREDICTION RESULTS WITH LOADING ANIMATION
    # ========================================================================
    if predict_button:
        # Show loading animation
        with st.spinner(''):
            st.markdown("""
            <div class="loading-container">
                <div class="spinner"></div>
                <div class="loading-text">🤖 Analyzing employee data...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate processing time for better UX
            time.sleep(1.5)
            
            # Create input DataFrame
            input_df = pd.DataFrame([input_data])[BEST_FEATURES]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            prob_stay = prediction_proba[0] * 100
            prob_leave = prediction_proba[1] * 100
        
        st.markdown("---")
        
        # Results Section
        if prediction == 0:
            st.markdown(f"""
            <div class="prediction-box stay-prediction">
                <div class="prediction-icon">✅</div>
                <h1>LIKELY TO STAY</h1>
                <p>This employee shows strong retention indicators</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box leave-prediction">
                <div class="prediction-icon">⚠️</div>
                <h1>AT RISK OF LEAVING</h1>
                <p>This employee shows turnover risk factors - consider intervention</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Probability Cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="probability-card">
                <div class="probability-label">Probability of Staying</div>
                <div class="probability-value prob-stay">{prob_stay:.1f}%</div>
                <div class="progress-bar-container">
                    <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
                    <div class="progress-text">{prob_stay:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="probability-card">
                <div class="probability-label">Probability of Leaving</div>
                <div class="probability-value prob-leave">{prob_leave:.1f}%</div>
                <div class="progress-bar-container">
                    <div class="progress-bar-red" style="width: {prob_leave}%;"></div>
                    <div class="progress-text">{prob_leave:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # INPUT SUMMARY
        # ====================================================================
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.expander("📋 View Input Summary", expanded=False):
                summary_df = pd.DataFrame({
                    'Feature': [
                        '😊 Satisfaction Level',
                        '📊 Last Evaluation',
                        '📅 Years at Company',
                        '📁 Number of Projects',
                        '⏰ Average Monthly Hours'
                    ],
                    'Value': [
                        f"{satisfaction_level:.2f}",
                        f"{last_evaluation:.2f}",
                        f"{time_spend_company} years",
                        f"{number_project} projects",
                        f"{average_monthly_hours} hours/month"
                    ]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Risk Assessment Tips
                st.markdown("---")
                st.markdown("### 💡 Retention Recommendations")
                
                if prediction == 1:  # At risk of leaving
                    if satisfaction_level < 0.5:
                        st.warning("⚠️ **Low Satisfaction**: Consider conducting a one-on-one discussion about job satisfaction and career goals.")
                    if average_monthly_hours > 250:
                        st.warning("⚠️ **High Workload**: Employee may be experiencing burnout. Review workload distribution.")
                    if number_project > 6:
                        st.warning("⚠️ **Too Many Projects**: Consider redistributing projects to prevent overwhelm.")
                    if time_spend_company >= 5:
                        st.info("💼 **Tenure**: Long-tenured employee at risk. Explore advancement opportunities.")
                else:  # Likely to stay
                    st.success("✅ **Positive Indicators**: Continue current engagement strategies and recognize contributions.")

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
