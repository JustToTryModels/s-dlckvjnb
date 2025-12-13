import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Employee Turnover Prediction | AI-Powered HR Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL CSS STYLING WITH ADVANCED ANIMATIONS & EFFECTS
# ============================================================================
st.markdown("""
<style>
    /* ===== IMPORT MODERN FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    /* ===== GLOBAL STYLES ===== */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* ===== GLASSMORPHISM CONTAINER ===== */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 3rem 2rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.1),
            0 0 1px rgba(255, 255, 255, 0.5) inset;
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ===== HEADER STYLES ===== */
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: slideDown 0.8s ease-out;
        letter-spacing: -1px;
        text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        animation: fadeIn 1s ease-out 0.3s both;
    }
    
    /* ===== MODERN CARD DESIGN ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px rgba(31, 38, 135, 0.15),
            0 0 1px rgba(255, 255, 255, 0.8) inset;
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.3),
            transparent
        );
        transition: left 0.5s ease;
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 20px 60px rgba(31, 38, 135, 0.25),
            0 0 1px rgba(255, 255, 255, 1) inset;
    }
    
    /* ===== FEATURE INPUT CARDS ===== */
    .feature-input-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 2px solid transparent;
        background-clip: padding-box;
        position: relative;
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .feature-input-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 16px;
        padding: 2px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .feature-input-card:hover::before {
        opacity: 1;
    }
    
    .feature-input-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    
    /* ===== PREDICTION RESULT CARDS ===== */
    .prediction-box {
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        animation: popIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }
    
    @keyframes popIn {
        0% { transform: scale(0.8); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .stay-prediction {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        position: relative;
    }
    
    .stay-prediction::before {
        content: '✓';
        position: absolute;
        font-size: 200px;
        opacity: 0.1;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .leave-prediction {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        position: relative;
    }
    
    .leave-prediction::before {
        content: '⚠';
        position: absolute;
        font-size: 200px;
        opacity: 0.1;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.1; }
        50% { transform: translate(-50%, -50%) scale(1.1); opacity: 0.15; }
    }
    
    .prediction-box h1 {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        position: relative;
        z-index: 1;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 0.8rem 0;
        position: relative;
        border-left: 5px solid;
        border-image: linear-gradient(180deg, #667eea, #764ba2) 1;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    
    /* ===== ENHANCED PROGRESS BARS WITH ANIMATION ===== */
    .progress-container {
        margin: 1.5rem 0;
    }
    
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #374151;
    }
    
    .progress-bar-container {
        width: 100%;
        height: 32px;
        background: #E5E7EB;
        border-radius: 16px;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .progress-bar-green {
        height: 100%;
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        border-radius: 16px;
        position: relative;
        transition: width 1.5s cubic-bezier(0.65, 0, 0.35, 1);
        animation: shimmer 2s infinite;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
    }
    
    .progress-bar-red {
        height: 100%;
        background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%);
        border-radius: 16px;
        position: relative;
        transition: width 1.5s cubic-bezier(0.65, 0, 0.35, 1);
        animation: shimmer 2s infinite;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    }
    
    @keyframes shimmer {
        0% { background-position: -100% 0; }
        100% { background-position: 200% 0; }
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
        animation: slide 2s infinite;
    }
    
    @keyframes slide {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* ===== VIBRANT RAINBOW GRADIENT PREDICT BUTTON - ENHANCED ===== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(
            45deg, 
            #667eea, #764ba2, #f093fb, #4facfe, #00f2fe, #667eea
        );
        background-size: 300% 300%;
        color: white !important;
        font-size: 1.5rem;
        font-weight: 900 !important;
        padding: 1.5rem 3rem;
        border-radius: 60px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 10px 40px rgba(102, 126, 234, 0.4),
            0 0 0 0 rgba(102, 126, 234, 0.5);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: gradientShift 4s ease infinite, buttonPulse 2s ease-in-out infinite;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes buttonPulse {
        0% { 
            box-shadow: 
                0 10px 40px rgba(102, 126, 234, 0.4),
                0 0 0 0 rgba(102, 126, 234, 0.5);
            transform: scale(1);
        }
        50% { 
            box-shadow: 
                0 15px 60px rgba(102, 126, 234, 0.6),
                0 0 0 15px rgba(102, 126, 234, 0);
            transform: scale(1.03);
        }
        100% { 
            box-shadow: 
                0 10px 40px rgba(102, 126, 234, 0.4),
                0 0 0 0 rgba(102, 126, 234, 0);
            transform: scale(1);
        }
    }
    
    .stButton>button:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 
            0 20px 60px rgba(102, 126, 234, 0.6),
            0 0 0 0 rgba(102, 126, 234, 0);
        animation: gradientShift 2s ease infinite;
    }
    
    .stButton>button:active {
        transform: translateY(-2px) scale(1.02);
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
            rgba(255, 255, 255, 0.4),
            transparent
        );
        transition: left 0.7s ease;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button::after {
        content: '🚀';
        position: absolute;
        font-size: 1.5rem;
        right: 30px;
        animation: rocket 2s ease-in-out infinite;
    }
    
    @keyframes rocket {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(10deg); }
    }
    
    /* ===== STYLED INPUTS ===== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #E5E7EB;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* ===== ENHANCED EXPANDER ===== */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stExpander"]:hover {
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    div[data-testid="stExpander"] details summary svg {
        color: white !important;
        fill: white !important;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1C3F 0%, #1a2744 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    section[data-testid="stSidebar"] * {
        color: #E5E7EB;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white;
        font-weight: 700;
    }
    
    /* ===== INFO BOX ===== */
    .info-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border: 2px solid #F59E0B;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.2);
    }
    
    /* ===== ICON BADGES ===== */
    .icon-badge {
        display: inline-block;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        line-height: 50px;
        font-size: 1.5rem;
        margin-right: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: iconFloat 3s ease-in-out infinite;
    }
    
    @keyframes iconFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid rgba(102, 126, 234, 0.2);
        color: #6B7280;
        font-size: 0.9rem;
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .prediction-box h1 {
            font-size: 2.5rem;
        }
        .glass-card {
            padding: 1.5rem;
        }
    }
    
    /* ===== ANIMATED COUNTERS ===== */
    .animated-counter {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: countUp 2s ease-out;
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ===== TOOLTIP ENHANCEMENT ===== */
    .tooltip-icon {
        cursor: help;
        color: #667eea;
        margin-left: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .tooltip-icon:hover {
        transform: scale(1.2);
        color: #764ba2;
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* ===== STAGGER ANIMATION FOR LIST ITEMS ===== */
    .stagger-item {
        animation: staggerFadeIn 0.5s ease-out backwards;
    }
    
    .stagger-item:nth-child(1) { animation-delay: 0.1s; }
    .stagger-item:nth-child(2) { animation-delay: 0.2s; }
    .stagger-item:nth-child(3) { animation-delay: 0.3s; }
    .stagger-item:nth-child(4) { animation-delay: 0.4s; }
    .stagger-item:nth-child(5) { animation-delay: 0.5s; }
    
    @keyframes staggerFadeIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
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
# CREATE CIRCULAR GAUGE CHART FOR PROBABILITY
# ============================================================================
def create_gauge_chart(probability, title, color):
    """Create a circular gauge chart using Plotly"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#374151'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': color},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': color,
            'steps': [
                {'range': [0, 33], 'color': '#FEE2E2'},
                {'range': [33, 66], 'color': '#FEF3C7'},
                {'range': [66, 100], 'color': '#D1FAE5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#374151", 'family': "Inter"},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

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
    # Header with animation
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered HR Analytics for Smarter Workforce Management</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown("## ℹ️ Model Information")
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>🤗 Repository:</strong><br>
            <code>{HF_REPO_ID}</code><br><br>
            <strong>🤖 Algorithm:</strong><br>
            Random Forest Classifier<br><br>
            <strong>📊 Features Used:</strong><br>
            {len(BEST_FEATURES)} features<br><br>
            <strong>⚖️ Class Weight:</strong><br>
            Balanced
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 Selected Features")
        feature_icons = ["😊", "📅", "⏰", "📁", "📊"]
        for i, (feat, icon) in enumerate(zip(BEST_FEATURES, feature_icons), 1):
            feat_display = feat.replace('_', ' ').title()
            st.markdown(f'<div class="stagger-item">{icon} {feat_display}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Target Variable")
        st.info("""
        **Prediction Classes:**
        - **0 → Stay**: Employee likely to stay
        - **1 → Leave**: Employee likely to leave
        """)
        
        st.markdown("---")
        
        st.markdown(f"[🔗 View on Hugging Face](https://huggingface.co/{HF_REPO_ID})")
        
        # Add timestamp
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # MAIN INPUT SECTION
    # ========================================================================
    st.markdown("---")
    st.markdown("## 📝 Enter Employee Information")
    
    st.markdown("""
    <div class="info-box">
        <strong>💡 Instructions:</strong> Fill in the employee's information below to generate an AI-powered turnover prediction.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # INPUT FORM IN GLASS CARDS
    # ========================================================================
    
    # ROW 1: Satisfaction & Evaluation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-input-card">', unsafe_allow_html=True)
        st.markdown("### 😊 Satisfaction Level")
        
        sat_col1, sat_col2 = st.columns([3, 1])
        with sat_col1:
            st.slider(
                "Satisfaction Slider",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.satisfaction_level,
                step=0.01,
                help="Employee satisfaction level (0 = Very Dissatisfied, 1 = Very Satisfied)",
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-input-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Last Evaluation Score")
        
        eval_col1, eval_col2 = st.columns([3, 1])
        with eval_col1:
            st.slider(
                "Evaluation Slider",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.last_evaluation,
                step=0.01,
                help="Last performance evaluation score (0 = Poor, 1 = Excellent)",
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ROW 2: Years & Projects
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="feature-input-card">', unsafe_allow_html=True)
        time_spend_company = st.number_input(
            "📅 Years at Company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            help="Number of years the employee has worked at the company"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="feature-input-card">', unsafe_allow_html=True)
        number_project = st.number_input(
            "📁 Number of Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of projects the employee is currently working on"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ROW 3: Monthly Hours
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown('<div class="feature-input-card">', unsafe_allow_html=True)
        average_monthly_hours = st.number_input(
            "⏰ Average Monthly Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            help="Average number of hours worked per month"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
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
        predict_button = st.button("🔮 Predict Employee Turnover", use_container_width=True)
    
    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================
    if predict_button:
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-box stay-prediction">
                    <h1>✅ WILL STAY</h1>
                    <p style="font-size: 1.4rem; margin-top: 1rem; position: relative; z-index: 1;">
                        Employee is <strong>likely to STAY</strong> with the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box leave-prediction">
                    <h1>⚠️ WILL LEAVE</h1>
                    <p style="font-size: 1.4rem; margin-top: 1rem; position: relative; z-index: 1;">
                        Employee is <strong>likely to LEAVE</strong> the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Confidence Levels")
            
            # Stay probability with enhanced progress bar
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-label">
                    <span>✅ Probability of Staying</span>
                    <span style="font-size: 1.2rem; color: #10B981;">{prob_stay:.1f}%</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leave probability with enhanced progress bar
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-label">
                    <span>⚠️ Probability of Leaving</span>
                    <span style="font-size: 1.2rem; color: #EF4444;">{prob_leave:.1f}%</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar-red" style="width: {prob_leave}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # CIRCULAR GAUGE CHARTS
        # ====================================================================
        st.markdown("---")
        st.markdown("## 📈 Visual Analytics")
        
        gauge_col1, gauge_col2 = st.columns(2)
        
        with gauge_col1:
            fig_stay = create_gauge_chart(prob_stay, "Probability to Stay", "#10B981")
            st.plotly_chart(fig_stay, use_container_width=True)
        
        with gauge_col2:
            fig_leave = create_gauge_chart(prob_leave, "Probability to Leave", "#EF4444")
            st.plotly_chart(fig_leave, use_container_width=True)
        
        # ====================================================================
        # RISK ASSESSMENT & RECOMMENDATIONS
        # ====================================================================
        st.markdown("---")
        st.markdown("## 💡 Risk Assessment & Recommendations")
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        if prediction == 1:  # High risk
            st.markdown("### ⚠️ High Turnover Risk Detected")
            st.markdown("""
            <div class="info-box" style="background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); border-color: #EF4444;">
                <strong>🚨 Recommended Actions:</strong>
                <ul>
                    <li>Schedule an immediate one-on-one meeting</li>
                    <li>Conduct a satisfaction survey to identify pain points</li>
                    <li>Review compensation and benefits package</li>
                    <li>Explore career development opportunities</li>
                    <li>Consider workload adjustments if needed</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Key factors analysis
            st.markdown("### 🔍 Key Factors Analysis")
            factors = []
            if satisfaction_level < 0.4:
                factors.append("❗ **Very Low Satisfaction** - Critical concern")
            elif satisfaction_level < 0.6:
                factors.append("⚠️ **Below Average Satisfaction** - Needs attention")
            
            if average_monthly_hours > 250:
                factors.append("❗ **Excessive Working Hours** - Risk of burnout")
            elif average_monthly_hours < 120:
                factors.append("⚠️ **Low Engagement** - Possible disengagement")
            
            if number_project > 6:
                factors.append("❗ **Overloaded with Projects** - May cause stress")
            elif number_project < 2:
                factors.append("⚠️ **Under-utilized** - May seek more challenges")
            
            if last_evaluation < 0.5:
                factors.append("❗ **Low Performance Rating** - May indicate issues")
            
            if time_spend_company < 2:
                factors.append("⚠️ **New Employee** - Higher turnover risk")
            elif time_spend_company > 6:
                factors.append("⚠️ **Long Tenure** - May be seeking change")
            
            for factor in factors:
                st.markdown(f"- {factor}")
        
        else:  # Low risk
            st.markdown("### ✅ Low Turnover Risk")
            st.markdown("""
            <div class="info-box" style="background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); border-color: #10B981;">
                <strong>👍 Positive Indicators:</strong>
                <ul>
                    <li>Employee shows strong engagement signals</li>
                    <li>Continue current retention strategies</li>
                    <li>Maintain regular check-ins and feedback</li>
                    <li>Explore opportunities for growth and development</li>
                    <li>Recognize and reward contributions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # INPUT SUMMARY
        # ====================================================================
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.expander("📋 View Detailed Input Summary", expanded=False):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                summary_data = {
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
                        f"{average_monthly_hours} hours"
                    ],
                    'Status': [
                        "🟢 Good" if satisfaction_level >= 0.6 else ("🟡 Fair" if satisfaction_level >= 0.4 else "🔴 Low"),
                        "🟢 Good" if last_evaluation >= 0.7 else ("🟡 Fair" if last_evaluation >= 0.5 else "🔴 Low"),
                        "🟢 Stable" if 2 <= time_spend_company <= 6 else "🟡 Monitor",
                        "🟢 Balanced" if 3 <= number_project <= 5 else "🟡 Check",
                        "🟢 Normal" if 150 <= average_monthly_hours <= 220 else "🟡 Review"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🤖 <strong>Powered by AI & Machine Learning</strong> | Built with Streamlit & Scikit-learn</p>
        <p>© 2024 Employee Turnover Prediction System | <a href="https://huggingface.co/Zlib2/RFC" target="_blank">View Model on Hugging Face</a></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
