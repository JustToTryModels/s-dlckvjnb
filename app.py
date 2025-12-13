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
    /* Center the main content with max-width - FROM CODE 2 */
    .block-container {
        max-width: 1200px !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
        margin: 0 auto !important;
    }
    
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
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1.5rem;
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
    
    /* INPUT CARDS - Light Cream Background */
    .feature-card {
        background-color: #FFE8C2;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #1E3A5F;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    /* ===== VIBRANT RAINBOW GRADIENT PREDICT BUTTON ===== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(
            45deg, 
            #ff0080, #ff8c00, #40e0d0, #ff0080, #ff8c00
        );
        background-size: 400% 400%;
        color: white !important;
        font-size: 1.4rem;
        font-weight: 900 !important;
        padding: 1.2rem 2.5rem;
        border-radius: 50px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 4px 15px rgba(255, 0, 128, 0.4),
            0 8px 30px rgba(255, 140, 0, 0.3),
            0 0 40px rgba(64, 224, 208, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: gradientShift 3s ease infinite, pulse 2s ease-in-out infinite;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Gradient animation - shifting colors */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Pulse glow animation */
    @keyframes pulse {
        0% { 
            box-shadow: 
                0 4px 15px rgba(255, 0, 128, 0.4),
                0 8px 30px rgba(255, 140, 0, 0.3),
                0 0 40px rgba(64, 224, 208, 0.2);
            transform: scale(1);
        }
        50% { 
            box-shadow: 
                0 6px 25px rgba(255, 0, 128, 0.6),
                0 12px 40px rgba(255, 140, 0, 0.5),
                0 0 60px rgba(64, 224, 208, 0.4),
                0 0 80px rgba(255, 0, 128, 0.2);
            transform: scale(1.02);
        }
        100% { 
            box-shadow: 
                0 4px 15px rgba(255, 0, 128, 0.4),
                0 8px 30px rgba(255, 140, 0, 0.3),
                0 0 40px rgba(64, 224, 208, 0.2);
            transform: scale(1);
        }
    }
    
    /* Hover - Electric effect with different gradient */
    .stButton>button:hover {
        background: linear-gradient(
            45deg, 
            #00f5ff, #ff00ff, #ffff00, #00f5ff, #ff00ff
        );
        background-size: 400% 400%;
        transform: translateY(-5px) scale(1.01);
        box-shadow: 
            0 10px 30px rgba(0, 245, 255, 0.5),
            0 15px 50px rgba(255, 0, 255, 0.4),
            0 0 100px rgba(255, 255, 0, 0.3),
            inset 0 0 20px rgba(255, 255, 255, 0.1);
        animation: gradientShift 1.5s ease infinite;
        color: white !important;
        border: none !important;
        outline: none !important;
        font-weight: 900 !important;
    }
    
    /* Active/Click - Neon burst effect */
    .stButton>button:active {
        background: linear-gradient(
            45deg, 
            #ff3366, #ff6b35, #f7931e, #ffd700, #ff3366
        );
        background-size: 400% 400%;
        transform: translateY(2px) scale(0.98);
        box-shadow: 
            0 2px 10px rgba(255, 51, 102, 0.6),
            0 4px 20px rgba(255, 107, 53, 0.4),
            inset 0 0 30px rgba(255, 255, 255, 0.2);
        color: white !important;
        border: none !important;
        outline: none !important;
        font-weight: 900 !important;
    }
    
    /* Shimmer/shine effect overlay */
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
    
    /* Sparkle particles effect */
    .stButton>button::after {
        content: '✨';
        position: absolute;
        font-size: 1.2rem;
        right: 20px;
        animation: sparkle 1.5s ease-in-out infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { opacity: 1; transform: scale(1) rotate(0deg); }
        50% { opacity: 0.5; transform: scale(1.3) rotate(180deg); }
    }
    
    /* Focus effect - Remove blue border completely */
    .stButton>button:focus {
        outline: none !important;
        border: none !important;
        box-shadow: 
            0 4px 15px rgba(255, 0, 128, 0.4),
            0 8px 30px rgba(255, 140, 0, 0.3),
            0 0 40px rgba(64, 224, 208, 0.2);
        color: white !important;
        font-weight: 900 !important;
    }
    
    /* Focus-visible - Remove blue border completely */
    .stButton>button:focus-visible {
        outline: none !important;
        border: none !important;
        box-shadow: 
            0 4px 15px rgba(255, 0, 128, 0.4),
            0 8px 30px rgba(255, 140, 0, 0.3),
            0 0 40px rgba(64, 224, 208, 0.2);
        font-weight: 900 !important;
    }
    
    /* Remove focus ring from button container as well */
    .stButton>button:focus:not(:focus-visible) {
        outline: none !important;
        border: none !important;
    }
    
    /* Ensure text stays white in ALL states */
    .stButton>button,
    .stButton>button:hover,
    .stButton>button:active,
    .stButton>button:focus,
    .stButton>button:focus-visible,
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
        outline: none !important;
        border: none !important;
        font-weight: 900 !important;
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
    # SIDEBAR (COPIED EXACTLY FROM CODE-1)
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
    st.markdown('<h2 class="section-header">📝 Enter Employee Information</h2>', unsafe_allow_html=True)
    
    # Initialize session state for syncing slider and number input
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # ROW 1: Satisfaction Level & Last Evaluation (side by side with feature cards)
    # ========================================================================
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">😊 <strong>Satisfaction Level</strong></span>
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
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">📊 <strong>Last Evaluation</strong></span>
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
        # ROW 2: Years at Company, Number of Projects, Average Monthly Hours (3 columns)
        # ========================================================================
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">📅 <strong>Years at Company</strong></span>
            </div>
            """, unsafe_allow_html=True)
            time_spend_company = st.number_input(
                "Years",
                min_value=1,
                max_value=40,
                value=3,
                step=1,
                label_visibility="collapsed",
                help="Number of years the employee has worked at the company"
            )
        
        with col4:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">📁 <strong>Number of Projects</strong></span>
            </div>
            """, unsafe_allow_html=True)
            number_project = st.number_input(
                "Projects",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                label_visibility="collapsed",
                help="Number of projects the employee is currently working on"
            )
        
        with col5:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">⏰ <strong>Avg. Monthly Hours</strong></span>
            </div>
            """, unsafe_allow_html=True)
            average_monthly_hours = st.number_input(
                "Hours",
                min_value=80,
                max_value=350,
                value=200,
                step=5,
                label_visibility="collapsed",
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
        predict_button = st.button("🔮 Predict Employee Turnover", use_container_width=True)
    
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

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
