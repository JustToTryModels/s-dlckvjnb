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
    page_title="Employee Turnover Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ENHANCED CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Animated Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ffffff, #e0e0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: headerFadeIn 1.5s ease-out;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    @keyframes headerFadeIn {
        0% { 
            opacity: 0;
            transform: translateY(-30px);
        }
        100% { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.95);
        text-align: center;
        margin-bottom: 2rem;
        animation: subHeaderFadeIn 1.5s ease-out 0.3s both;
        font-weight: 400;
    }
    
    @keyframes subHeaderFadeIn {
        0% { 
            opacity: 0;
            transform: translateY(-20px);
        }
        100% { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Glass Morphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        0% { 
            opacity: 0;
            transform: translateY(30px);
        }
        100% { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Prediction Box with Animation */
    .prediction-box {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        animation: predictionReveal 0.8s ease-out;
    }
    
    @keyframes predictionReveal {
        0% { 
            opacity: 0;
            transform: scale(0.9);
        }
        100% { 
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .stay-prediction {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        box-shadow: 0 10px 40px rgba(132, 250, 176, 0.4);
    }
    
    .leave-prediction {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        box-shadow: 0 10px 40px rgba(250, 112, 154, 0.4);
    }
    
    .prediction-box h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
        animation: bounceIn 0.8s ease-out 0.3s both;
    }
    
    @keyframes bounceIn {
        0% { 
            opacity: 0;
            transform: scale(0.3);
        }
        50% { 
            transform: scale(1.05);
        }
        100% { 
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Enhanced Input Containers */
    .input-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .input-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    
    /* Enhanced Progress Bars */
    .progress-bar-container {
        width: 100%;
        background-color: rgba(0, 0, 0, 0.1);
        border-radius: 50px;
        margin: 10px 0;
        height: 30px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-bar-green {
        height: 100%;
        background: linear-gradient(90deg, #84fab0, #8fd3f4);
        border-radius: 50px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar-red {
        height: 100%;
        background: linear-gradient(90deg, #fa709a, #fee140);
        border-radius: 50px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
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
    
    /* Enhanced Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
        backdrop-filter: blur(10px);
    }
    
    .sidebar-info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced Button with Loading State */
    .stButton>button {
        width: 100%;
        background: linear-gradient(
            45deg, 
            #667eea, #764ba2, #f093fb, #667eea
        );
        background-size: 400% 400%;
        color: white !important;
        font-size: 1.3rem;
        font-weight: 700 !important;
        padding: 1.2rem 2.5rem;
        border-radius: 50px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: gradientShift 4s ease infinite;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
        background: linear-gradient(
            45deg, 
            #f093fb, #f5576c, #4facfe, #f093fb
        );
        background-size: 400% 400%;
        animation: gradientShift 1.5s ease infinite;
    }
    
    .stButton>button:active {
        transform: translateY(1px) scale(0.98);
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Enhanced Expander */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 15px !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: scale(1.02);
    }
    
    /* Feature Icons */
    .feature-icon {
        display: inline-block;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        color: white;
        font-size: 1.2rem;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
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
        font-size: 0.9rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
        }
        
        .glass-card {
            padding: 1.5rem;
        }
        
        .prediction-box {
            padding: 1.5rem;
        }
        
        .prediction-box h1 {
            font-size: 2rem;
        }
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
    """Sync satisfaction level from slider to session state"""
    st.session_state.satisfaction_level = st.session_state.sat_slider

def sync_satisfaction_input():
    """Sync satisfaction level from number input to session state"""
    st.session_state.satisfaction_level = st.session_state.sat_input

def sync_evaluation_slider():
    """Sync evaluation from slider to session state"""
    st.session_state.last_evaluation = st.session_state.eval_slider

def sync_evaluation_input():
    """Sync evaluation from number input to session state"""
    st.session_state.last_evaluation = st.session_state.eval_input

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header with animation
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🔮 Advanced AI-Powered Employee Retention Analytics</p>', unsafe_allow_html=True)
    
    # Load model silently
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    
    # ========================================================================
    # ENHANCED SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-info-card">
            <h3 style="margin-top: 0;">🤖 Model Information</h3>
            <p><strong>Repository:</strong> Zlib2/RFC</p>
            <p><strong>Algorithm:</strong> Random Forest</p>
            <p><strong>Features:</strong> 5 Key Metrics</p>
            <p><strong>Accuracy:</strong> 95%+</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📊 Key Features Used")
        feature_icons = ["😊", "📅", "⏰", "📁", "📊"]
        for i, (feat, icon) in enumerate(zip(BEST_FEATURES, feature_icons), 1):
            feat_display = feat.replace('_', ' ').title()
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 10px 0;">
                <span class="feature-icon">{icon}</span>
                <span>{i}. {feat_display}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="background: #f0f2f6; padding: 1rem; border-radius: 10px;">
            <h4>🎯 Prediction Classes</h4>
            <p style="margin: 5px 0;"><span style="color: #28a745;">●</span> <strong>Stay</strong>: Employee likely to remain</p>
            <p style="margin: 5px 0;"><span style="color: #dc3545;">●</span> <strong>Leave</strong>: Employee likely to depart</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank" 
               style="display: inline-block; padding: 10px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; text-decoration: none; border-radius: 25px; font-weight: 600;">
                🔗 View on Hugging Face
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN INPUT SECTION WITH GLASS MORPHISM
    # ========================================================================
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Employee Information")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea15, #764ba215); padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea;">
        <strong>💡 Tip:</strong> Provide accurate employee data for the most reliable predictions. Our AI model analyzes key factors that influence employee retention.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for syncing slider and number input
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # ROW 1: Satisfaction Level & Last Evaluation (side by side)
    # ========================================================================
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("#### 😊 Satisfaction Level")
        
        col1, col2 = st.columns([3, 1])
        with col1:
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
        with col2:
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
        st.markdown('</div>', unsafe_allow_html=True)
        
        satisfaction_level = st.session_state.satisfaction_level
    
    with row1_col2:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 Last Evaluation Score")
        
        col1, col2 = st.columns([3, 1])
        with col1:
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
        with col2:
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
        st.markdown('</div>', unsafe_allow_html=True)
        
        last_evaluation = st.session_state.last_evaluation
    
    # ========================================================================
    # ROW 2: Years at Company & Number of Projects (side by side)
    # ========================================================================
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        time_spend_company = st.number_input(
            "📅 Years at Company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            help="Number of years the employee has worked at the company"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with row2_col2:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        number_project = st.number_input(
            "📁 Number of Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of projects the employee is currently working on"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # ROW 3: Average Monthly Hours
    # ========================================================================
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close glass card
    
    # ========================================================================
    # PREDICTION BUTTON WITH LOADING STATE
    # ========================================================================
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if 'predicting' not in st.session_state:
            st.session_state.predicting = False
        
        predict_button = st.button(
            "🔮 Predict Employee Turnover",
            use_container_width=True,
            disabled=st.session_state.predicting
        )
    
    # ========================================================================
    # PREDICTION RESULTS WITH ENHANCED VISUALIZATION
    # ========================================================================
    if predict_button and not st.session_state.predicting:
        st.session_state.predicting = True
        
        # Show loading animation
        with st.spinner('🤖 Analyzing employee data...'):
            time.sleep(1.5)  # Simulate processing time
            
            # Create input DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])[BEST_FEATURES]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            prob_stay = prediction_proba[0] * 100
            prob_leave = prediction_proba[1] * 100
        
        st.session_state.predicting = False
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Results")
        
        # Results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-box stay-prediction">
                    <h1>✅ STAY</h1>
                    <p style="font-size: 1.2rem; margin-top: 1rem; font-weight: 600;">
                        Employee is likely to <strong>REMAIN</strong> with the company
                    </p>
                    <p style="margin-top: 1rem; opacity: 0.9;">
                        🎉 Great news! This employee shows strong retention indicators
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box leave-prediction">
                    <h1>⚠️ LEAVE</h1>
                    <p style="font-size: 1.2rem; margin-top: 1rem; font-weight: 600;">
                        Employee is likely to <strong>DEPART</strong> from the company
                    </p>
                    <p style="margin-top: 1rem; opacity: 0.9;">
                        ⚡ Consider retention strategies to address potential turnover
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Confidence Analysis")
            
            # Stay probability with enhanced visualization
            st.markdown(f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-weight: 600; color: #28a745;">✅ Stay Probability</span>
                    <span style="font-weight: 700; font-size: 1.2rem; color: #28a745;">{prob_stay:.1f}%</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leave probability with enhanced visualization
            st.markdown(f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-weight: 600; color: #dc3545;">⚠️ Leave Probability</span>
                    <span style="font-weight: 700; font-size: 1.2rem; color: #dc3545;">{prob_leave:.1f}%</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar-red" style="width: {prob_leave}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence indicator
            confidence = max(prob_stay, prob_leave)
            confidence_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
            confidence_color = "#28a745" if confidence > 80 else "#ffc107" if confidence > 60 else "#dc3545"
            
            st.markdown(f"""
            <div style="background: {confidence_color}20; padding: 1rem; border-radius: 10px; margin-top: 20px; border-left: 4px solid {confidence_color};">
                <strong>🎯 Model Confidence:</strong> 
                <span style="color: {confidence_color}; font-weight: 700;">{confidence_level}</span>
                <br>
                <small style="opacity: 0.8;">Based on {confidence:.1f}% prediction certainty</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # ENHANCED INPUT SUMMARY (Collapsible Dropdown)
        # ====================================================================
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.expander("📋 View Input Summary & Analysis", expanded=False):
                # Summary table with icons
                summary_data = {
                    'Metric': [
                        '😊 Satisfaction Level',
                        '📊 Last Evaluation',
                        '📅 Years at Company',
                        '📁 Number of Projects',
                        '⏰ Average Monthly Hours'
                    ],
                    'Value': [
                        f"{satisfaction_level:.2f} ({'High' if satisfaction_level > 0.7 else 'Medium' if satisfaction_level > 0.4 else 'Low'})",
                        f"{last_evaluation:.2f} ({'Excellent' if last_evaluation > 0.8 else 'Good' if last_evaluation > 0.6 else 'Needs Improvement'})",
                        f"{time_spend_company} years ({'Veteran' if time_spend_company > 5 else 'Experienced' if time_spend_company > 2 else 'New'})",
                        f"{number_project} projects ({'High Load' if number_project > 6 else 'Normal' if number_project > 3 else 'Light'})",
                        f"{average_monthly_hours} hrs/month ({'Overtime' if average_monthly_hours > 220 else 'Standard' if average_monthly_hours > 160 else 'Part-time'})"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Key insights
                st.markdown("### 🔍 Key Insights")
                
                insights = []
                if satisfaction_level < 0.5:
                    insights.append("⚠️ Low satisfaction level may increase turnover risk")
                if time_spend_company > 5 and satisfaction_level < 0.6:
                    insights.append("📅 Long tenure with low satisfaction is a red flag")
                if average_monthly_hours > 250:
                    insights.append("⏰ High workload may lead to burnout")
                if number_project > 6:
                    insights.append("📁 High project count indicates potential overload")
                if last_evaluation > 0.8 and satisfaction_level < 0.5:
                    insights.append("📊 High performer with low satisfaction needs attention")
                
                if insights:
                    for insight in insights:
                        st.markdown(f"""
                        <div style="background: #fff3cd; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #ffc107;">
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #d4edda; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #28a745;">
                        ✅ Employee metrics appear balanced and healthy
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
