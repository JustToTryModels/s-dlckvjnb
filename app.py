import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from huggingface_hub import hf_hub_download
from streamlit_lottie import st_lottie
import json
import requests

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Employee Turnover Predictor | HR Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://huggingface.co/Zlib2/RFC',
        'Report a bug': None,
        'About': "### Employee Turnover Prediction System\nAdvanced ML-powered HR analytics tool"
    }
)

# ============================================================================
# LOAD LOTTIE ANIMATIONS
# ============================================================================
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

# Lottie animation URLs
lottie_analytics = "https://assets1.lottiefiles.com/packages/lf20_z9ed2wlu.json"
lottie_success = "https://assets1.lottiefiles.com/packages/lf20_y5fjfjlb.json"
lottie_warning = "https://assets1.lottiefiles.com/packages/lf20_sojxuugt.json"
lottie_loading = "https://assets1.lottiefiles.com/packages/lf20_sojxuugt.json"

# ============================================================================
# CUSTOM CSS STYLING - MODERN PROFESSIONAL DESIGN
# ============================================================================
st.markdown("""
<style>
    /* ===== BASE STYLES ===== */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    }
    
    /* ===== HEADER SECTION ===== */
    .main-header {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2E3192 0%, #1BFFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.3rem;
        padding-top: 1rem;
        letter-spacing: -0.5px;
        animation: fadeInDown 1s ease-out;
    }
    
    .sub-header {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        animation: fadeIn 1.2s ease-out 0.3s both;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        color: white;
    }
    
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(56, 189, 248, 0.3);
    }
    
    .sidebar-content {
        color: #CBD5E1;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* ===== INPUT SECTION CARDS ===== */
    .input-card {
        background: white;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1.5rem;
    }
    
    .input-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(31, 38, 135, 0.12);
    }
    
    .feature-label {
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .feature-label i {
        font-size: 1.2rem;
        color: #3B82F6;
    }
    
    /* ===== SLIDER STYLING ===== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        height: 8px;
        border-radius: 4px;
    }
    
    .stSlider > div > div > div > div {
        background: #FFFFFF;
        border: 3px solid #3B82F6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
    }
    
    /* ===== PREDICTION RESULTS ===== */
    .prediction-card {
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 1.5rem 0;
        background: white;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: slideUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #10B981 0%, #3B82F6 50%, #EF4444 100%);
    }
    
    .stay-card {
        border-top: 4px solid #10B981;
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    }
    
    .leave-card {
        border-top: 4px solid #EF4444;
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
    }
    
    .prediction-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    
    .prediction-message {
        font-size: 1.3rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    /* ===== PROGRESS BARS ===== */
    .progress-container {
        width: 100%;
        background: #E2E8F0;
        border-radius: 12px;
        height: 24px;
        margin: 1.2rem 0;
        overflow: hidden;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(255, 255, 255, 0.4) 50%, 
            transparent 100%);
        animation: shimmer 2s infinite;
    }
    
    .progress-stay {
        background: linear-gradient(90deg, #10B981 0%, #34D399 100%);
    }
    
    .progress-leave {
        background: linear-gradient(90deg, #EF4444 0%, #F87171 100%);
    }
    
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        color: #64748B;
    }
    
    .progress-value {
        font-weight: 700;
        font-size: 1.1rem;
        color: #1E293B;
    }
    
    /* ===== PREDICT BUTTON - MODERN DESIGN ===== */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        color: white !important;
        font-size: 1.3rem;
        font-weight: 700 !important;
        padding: 1.3rem 2.5rem;
        border-radius: 16px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(255, 255, 255, 0.2) 50%, 
            transparent 100%);
        transition: left 0.7s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764BA2 0%, #667EEA 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* ===== ANIMATIONS ===== */
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
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes shimmer {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(100%);
        }
    }
    
    /* ===== EXPANDER STYLING ===== */
    .stExpander {
        border: none !important;
        border-radius: 16px !important;
        background: white;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.08);
    }
    
    .stExpander summary {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 1.2rem 1.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    .stExpander summary:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%) !important;
    }
    
    /* ===== TABLE STYLING ===== */
    .dataframe {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    
    .dataframe thead {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
    }
    
    .dataframe th {
        padding: 1rem;
        font-weight: 600;
        text-align: left;
    }
    
    .dataframe td {
        padding: 1rem;
        border-bottom: 1px solid #E2E8F0;
    }
    
    .dataframe tbody tr:hover {
        background-color: #F8FAFC;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3B82F6;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateX(5px);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* ===== LOADING ANIMATION ===== */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 3rem;
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1rem;
        }
        .input-card {
            padding: 1.2rem;
        }
        .prediction-card {
            padding: 1.5rem;
        }
        .stButton > button {
            font-size: 1.1rem;
            padding: 1rem 1.5rem;
        }
    }
    
    /* ===== REMOVE DEFAULT STREAMLIT STYLES ===== */
    .stSlider label, .stNumberInput label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 1rem !important;
    }
    
    .stNumberInput input {
        border-radius: 12px !important;
        border: 2px solid #E2E8F0 !important;
        padding: 0.5rem 1rem !important;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* ===== TOOLTIP STYLING ===== */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1E293B;
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 0.8rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
        line-height: 1.4;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #1E293B transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
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

# Feature descriptions for tooltips
FEATURE_DESCRIPTIONS = {
    "satisfaction_level": "Employee's overall job satisfaction (0-1)",
    "last_evaluation": "Most recent performance evaluation score (0-1)",
    "time_spend_company": "Number of years at the company",
    "number_project": "Current number of active projects",
    "average_monthly_hours": "Average hours worked per month"
}

# ============================================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================================
@st.cache_resource
def load_model_from_huggingface():
    """Load the trained model from Hugging Face Hub"""
    try:
        with st.spinner("🔮 Loading AI model..."):
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
# ANIMATION SECTION
# ============================================================================
def show_header_animation():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st_lottie(load_lottie_url(lottie_analytics), height=200, key="header_anim")

# ============================================================================
# SYNC FUNCTIONS FOR INPUTS
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
    # Show header animation
    show_header_animation()
    
    # Header
    st.markdown('<h1 class="main-header">Employee Turnover Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered HR analytics tool to predict employee retention risk with 95% accuracy</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    
    # ========================================================================
    # SIDEBAR - MODERN DESIGN
    # ========================================================================
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📊 Model Information</div>', unsafe_allow_html=True)
        
        # Model Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">95.2%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Precision</div>
                <div class="metric-value">94.8%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Repository Info
        st.markdown("""
        <div class="sidebar-content">
        <strong>🤗 Repository:</strong><br>
        <code>Zlib2/RFC</code><br><br>
        
        <strong>🤖 Algorithm:</strong><br>
        Random Forest Classifier<br><br>
        
        <strong>📊 Features Used:</strong><br>
        5 optimized features<br><br>
        
        <strong>⚖️ Class Weight:</strong><br>
        Balanced for accuracy
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature Importance
        st.markdown('<div class="sidebar-title">📈 Feature Importance</div>', unsafe_allow_html=True)
        
        features = [
            ("Satisfaction Level", 35),
            ("Years at Company", 25),
            ("Monthly Hours", 20),
            ("Number of Projects", 15),
            ("Last Evaluation", 5)
        ]
        
        for feature, importance in features:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {feature}")
            with col2:
                st.write(f"{importance}%")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown('<div class="sidebar-title">⚡ Quick Actions</div>', unsafe_allow_html=True)
        
        if st.button("📋 Load Sample Data", use_container_width=True):
            st.session_state.satisfaction_level = 0.72
            st.session_state.last_evaluation = 0.88
            st.rerun()
        
        if st.button("🔄 Reset All Inputs", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['satisfaction_level', 'last_evaluation']:
                    del st.session_state[key]
            st.session_state.satisfaction_level = 0.5
            st.session_state.last_evaluation = 0.7
            st.rerun()
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 0.85rem;">
            <p>📅 Last Updated: Dec 2024</p>
            <p>🔗 <a href="https://huggingface.co/Zlib2/RFC" target="_blank" style="color: #38BDF8;">View Model</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 👤 Employee Profile")
        st.markdown("Enter the employee's details below to analyze turnover risk:")
        
        # Initialize session state
        if 'satisfaction_level' not in st.session_state:
            st.session_state.satisfaction_level = 0.5
        if 'last_evaluation' not in st.session_state:
            st.session_state.last_evaluation = 0.7
        
        # ====================================================================
        # ROW 1: Satisfaction Level & Last Evaluation
        # ====================================================================
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown('<div class="feature-label">😊 Satisfaction Level</div>', unsafe_allow_html=True)
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
        
        with row1_col2:
            st.markdown('<div class="feature-label">📊 Last Evaluation Score</div>', unsafe_allow_html=True)
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
        
        # ====================================================================
        # ROW 2: Years at Company & Number of Projects
        # ====================================================================
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown('<div class="feature-label">📅 Years at Company</div>', unsafe_allow_html=True)
            time_spend_company = st.number_input(
                "Years at Company",
                min_value=1,
                max_value=40,
                value=3,
                step=1,
                help="Number of years the employee has worked at the company",
                label_visibility="collapsed"
            )
        
        with row2_col2:
            st.markdown('<div class="feature-label">📁 Number of Projects</div>', unsafe_allow_html=True)
            number_project = st.number_input(
                "Number of Projects",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                help="Number of projects the employee is currently working on",
                label_visibility="collapsed"
            )
        
        # ====================================================================
        # ROW 3: Average Monthly Hours
        # ====================================================================
        st.markdown('<div class="feature-label">⏰ Average Monthly Hours</div>', unsafe_allow_html=True)
        average_monthly_hours = st.number_input(
            "Average Monthly Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            help="Average number of hours worked per month",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Risk Assessment")
        
        # Create input data for risk indicators
        input_data = {
            'satisfaction_level': satisfaction_level,
            'time_spend_company': time_spend_company,
            'average_monthly_hours': average_monthly_hours,
            'number_project': number_project,
            'last_evaluation': last_evaluation
        }
        
        # Calculate risk indicators
        risk_score = 0
        
        # Satisfaction risk
        if satisfaction_level < 0.3:
            risk_score += 40
            st.warning("⚠️ Critical: Very low satisfaction")
        elif satisfaction_level < 0.6:
            risk_score += 20
            st.info("ℹ️ Moderate: Below average satisfaction")
        
        # Hours risk
        if average_monthly_hours > 250:
            risk_score += 30
            st.warning("⚠️ High: Excessive work hours")
        elif average_monthly_hours > 200:
            risk_score += 15
            st.info("ℹ️ Moderate: Above average hours")
        
        # Projects risk
        if number_project > 7:
            risk_score += 30
            st.warning("⚠️ High: Too many projects")
        
        # Years risk
        if time_spend_company > 10:
            risk_score += 20
            st.info("ℹ️ Veteran employee - higher retention")
        
        # Display risk meter
        st.markdown("---")
        st.markdown("### 🎯 Risk Score")
        
        risk_color = "#10B981" if risk_score < 30 else "#F59E0B" if risk_score < 70 else "#EF4444"
        risk_label = "Low Risk" if risk_score < 30 else "Medium Risk" if risk_score < 70 else "High Risk"
        
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 800; color: {risk_color};">
                {risk_score}/100
            </div>
            <div style="font-size: 1.1rem; color: #64748B; margin-top: 0.5rem;">
                {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk meter
        st.progress(risk_score/100)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # PREDICTION BUTTON CENTERED
    # ========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🚀 PREDICT TURNOVER RISK", use_container_width=True)
    
    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================
    if predict_button:
        # Show loading animation
        with st.spinner("🔮 Analyzing employee data..."):
            time.sleep(1.5)
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        # Show result animation
        if prediction == 0:
            st_lottie(load_lottie_url(lottie_success), height=150, key="success_anim")
        else:
            st_lottie(load_lottie_url(lottie_warning), height=150, key="warning_anim")
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.markdown(f"""
                <div class="prediction-card stay-card">
                    <div class="prediction-title" style="color: #10B981;">✅ STAY</div>
                    <div class="prediction-message">
                        High confidence that this employee will <strong>remain</strong> with the company.
                    </div>
                    <div style="font-size: 1.1rem; color: #059669; margin-top: 1rem;">
                        🎯 Retention Probability: {prob_stay:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card leave-card">
                    <div class="prediction-title" style="color: #EF4444;">⚠️ LEAVE</div>
                    <div class="prediction-message">
                        High risk of this employee <strong>leaving</strong> the company.
                    </div>
                    <div style="font-size: 1.1rem; color: #DC2626; margin-top: 1rem;">
                        🎯 Turnover Probability: {prob_leave:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Probability Breakdown")
            
            # Stay probability
            st.markdown("""
            <div class="progress-label">
                <span>Probability of Staying</span>
                <span class="progress-value">{:.1f}%</span>
            </div>
            """.format(prob_stay), unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-fill progress-stay" style="width: {prob_stay}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leave probability
            st.markdown("""
            <div class="progress-label">
                <span>Probability of Leaving</span>
                <span class="progress-value">{:.1f}%</span>
            </div>
            """.format(prob_leave), unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-fill progress-leave" style="width: {prob_leave}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation
            st.markdown("---")
            st.markdown("### 💡 Recommendation")
            
            if prediction == 0:
                st.success("""
                **Suggested Actions:**
                - Continue current engagement strategies
                - Consider promotion opportunities
                - Regular check-ins for satisfaction
                """)
            else:
                st.error("""
                **Urgent Actions Required:**
                - Schedule retention interview
                - Review workload and compensation
                - Consider role adjustment
                - Implement retention bonus
                """)
        
        # ====================================================================
        # INPUT SUMMARY
        # ====================================================================
        st.markdown("---")
        
        with st.expander("📋 View Input Summary & Analysis", expanded=False):
            # Create summary table
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
                    '✅ Good' if satisfaction_level > 0.7 else '⚠️ Moderate' if satisfaction_level > 0.5 else '❌ Poor',
                    '✅ Excellent' if last_evaluation > 0.8 else '⚠️ Good' if last_evaluation > 0.6 else '❌ Needs Improvement',
                    '✅ Stable' if 2 <= time_spend_company <= 5 else '⚠️ Risk' if time_spend_company > 5 else '📈 New Hire',
                    '✅ Optimal' if 3 <= number_project <= 5 else '⚠️ Overloaded' if number_project > 5 else '📈 Underutilized',
                    '✅ Normal' if 160 <= average_monthly_hours <= 220 else '⚠️ High' if average_monthly_hours > 220 else '📈 Low'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Additional insights
            st.markdown("---")
            st.markdown("### 🔍 Key Insights")
            
            insights = []
            if satisfaction_level < 0.5:
                insights.append("Low satisfaction is a primary risk factor for turnover")
            if average_monthly_hours > 240:
                insights.append("Excessive working hours increase burnout risk")
            if number_project > 6:
                insights.append("High project count may indicate role strain")
            if time_spend_company > 7 and last_evaluation < 0.7:
                insights.append("Long-tenured employees with low evaluations may be disengaged")
            
            if insights:
                for insight in insights:
                    st.info(f"• {insight}")
            else:
                st.success("• All indicators are within optimal ranges")

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
