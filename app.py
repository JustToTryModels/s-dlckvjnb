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
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .stay-prediction {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .leave-prediction {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .feature-box {
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #b8daff;
    }
    
    /* ===== ATTRACTIVE PREDICT BUTTON STYLING - DARKER PROFESSIONAL THEME ===== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 25%, #0f3460 50%, #1a1a2e 75%, #16213e 100%);
        background-size: 300% 300%;
        color: white !important;
        font-size: 1.3rem;
        font-weight: bold;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: 2px solid #4a90d9;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(74, 144, 217, 0.3), 0 8px 30px rgba(26, 26, 46, 0.4);
        transition: all 0.3s ease;
        animation: darkGradient 4s ease infinite, subtleGlow 2.5s ease-in-out infinite;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Dark gradient animation */
    @keyframes darkGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Subtle glow animation */
    @keyframes subtleGlow {
        0% { 
            box-shadow: 0 4px 15px rgba(74, 144, 217, 0.3), 0 8px 30px rgba(26, 26, 46, 0.4);
            border-color: #4a90d9;
        }
        50% { 
            box-shadow: 0 4px 20px rgba(74, 144, 217, 0.5), 0 8px 40px rgba(26, 26, 46, 0.5), 0 0 30px rgba(74, 144, 217, 0.2);
            border-color: #6ba3e0;
        }
        100% { 
            box-shadow: 0 4px 15px rgba(74, 144, 217, 0.3), 0 8px 30px rgba(26, 26, 46, 0.4);
            border-color: #4a90d9;
        }
    }
    
    /* Hover effects */
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.02);
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #1a1a2e 100%);
        background-size: 300% 300%;
        box-shadow: 0 8px 25px rgba(74, 144, 217, 0.5), 0 12px 40px rgba(26, 26, 46, 0.5), 0 0 50px rgba(74, 144, 217, 0.3);
        border-color: #7bb3e8;
        color: white !important;
    }
    
    /* Active/Click effect */
    .stButton>button:active {
        transform: translateY(2px) scale(0.98);
        box-shadow: 0 2px 10px rgba(74, 144, 217, 0.4), 0 4px 20px rgba(26, 26, 46, 0.3);
        color: white !important;
    }
    
    /* Shimmer effect overlay */
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.15),
            transparent
        );
        transition: left 0.6s ease;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Focus effect */
    .stButton>button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(74, 144, 217, 0.4), 0 4px 20px rgba(74, 144, 217, 0.5), 0 8px 30px rgba(26, 26, 46, 0.4);
        color: white !important;
    }
    
    /* Smooth hover animation */
    .stButton>button:hover {
        animation: darkGradient 2s ease infinite;
    }
    
    /* Ensure text stays white in ALL states */
    .stButton>button,
    .stButton>button:hover,
    .stButton>button:active,
    .stButton>button:focus,
    .stButton>button:visited,
    .stButton>button span,
    .stButton>button:hover span,
    .stButton>button:active span,
    .stButton>button:focus span,
    .stButton>button p,
    .stButton>button:hover p,
    .stButton>button:active p,
    .stButton>button:focus p,
    .stButton>button div,
    .stButton>button:hover div,
    .stButton>button:active div,
    .stButton>button:focus div {
        color: white !important;
    }
    
    .info-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .progress-bar-container {
        width: 100%;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 5px 0 15px 0;
        height: 25px;
        overflow: hidden;
    }
    .progress-bar-green {
        height: 100%;
        background-color: #28a745;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    .progress-bar-red {
        height: 100%;
        background-color: #dc3545;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Blue styled expander */
    div[data-testid="stExpander"] {
        border: none !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details {
        border: none !important;
    }
    div[data-testid="stExpander"] details summary {
        background-color: #1E3A5F !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        background-color: #2E5A8F !important;
        color: white !important;
    }
    div[data-testid="stExpander"] details summary svg {
        color: white !important;
        fill: white !important;
    }
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 8px 8px 0 0 !important;
    }
    div[data-testid="stExpander"] details > div {
        border: 1px solid #1E3A5F !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
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
    # Header
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether an employee is likely to leave the company</p>', unsafe_allow_html=True)
    
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
        st.header("ℹ️ Model Information")
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>🤗 Repository:</strong><br>{HF_REPO_ID}<br><br>
            <strong>🤖 Algorithm:</strong><br>Random Forest Classifier<br><br>
            <strong>📊 Features Used:</strong><br>{len(BEST_FEATURES)} features<br><br>
            <strong>⚖️ Class Weight:</strong><br>Balanced
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📋 Selected Features")
        for i, feat in enumerate(BEST_FEATURES, 1):
            feat_display = feat.replace('_', ' ').title()
            st.write(f"{i}. {feat_display}")
        
        st.markdown("---")
        
        st.subheader("🎯 Target Variable")
        st.info("""
        **Prediction Classes:**
        - **0 → Stay**: Employee likely to stay
        - **1 → Leave**: Employee likely to leave
        """)
        
        st.markdown("---")
        
        st.markdown(f"[🔗 View on Hugging Face](https://huggingface.co/{HF_REPO_ID})")
    
    # ========================================================================
    # MAIN INPUT SECTION
    # ========================================================================
    st.markdown("---")
    st.subheader("📝 Enter Employee Information")
    
    st.markdown("""
    <div class="feature-box">
        <strong>📌 Instructions:</strong> Enter the employee's information below to predict turnover risk.
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
        # Satisfaction Level with both slider and number input
        st.write("😊 **Satisfaction Level**")
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
        # Last Evaluation with both slider and number input
        st.write("📊 **Last Evaluation Score**")
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
    
    # ========================================================================
    # ROW 2: Years at Company & Number of Projects (side by side)
    # ========================================================================
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Time Spent at Company
        time_spend_company = st.number_input(
            "📅 Years at Company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            help="Number of years the employee has worked at the company"
        )
    
    with row2_col2:
        # Number of Projects
        number_project = st.number_input(
            "📁 Number of Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of projects the employee is currently working on"
        )
    
    # ========================================================================
    # ROW 3: Average Monthly Hours
    # ========================================================================
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        # Average Monthly Hours
        average_monthly_hours = st.number_input(
            "⏰ Average Monthly Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            help="Average number of hours worked per month"
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
        predict_button = st.button("🔮 𝐏𝐫𝐞𝐝𝐢𝐜𝐭 𝐄𝐦𝐩𝐥𝐨𝐲𝐞𝐞 𝐓𝐮𝐫𝐧𝐨𝐯𝐞𝐫", use_container_width=True)
    
    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================
    if predict_button:
        # Create input DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-box stay-prediction">
                    <h1>✅ STAY</h1>
                    <p style="font-size: 1.3rem; margin-top: 1rem;">
                        Employee is likely to <strong>STAY</strong> with the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box leave-prediction">
                    <h1>⚠️ LEAVE</h1>
                    <p style="font-size: 1.3rem; margin-top: 1rem;">
                        Employee is likely to <strong>LEAVE</strong> the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Prediction Probabilities")
            
            # Stay probability with GREEN bar
            st.write(f"**Probability of Staying:** {prob_stay:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leave probability with RED bar
            st.write(f"**Probability of Leaving:** {prob_leave:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-red" style="width: {prob_leave}%;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # INPUT SUMMARY (Collapsible Dropdown) - Centered like predict button
        # ====================================================================
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.expander("📋 Click to View Input Summary", expanded=False):
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
                        f"{average_monthly_hours} hours"
                    ]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
