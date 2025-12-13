import streamlit as st
import joblib
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
    /* MAIN BACKGROUND & FONTS */
    .main {
        background-color: #fcfcfc;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #1E3A5F;
    }
    
    /* HEADER STYLES */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeIn 1.5s ease-out;
    }

    /* ANIMATIONS */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(255, 0, 128, 0.4); transform: scale(1); }
        50% { box-shadow: 0 6px 25px rgba(255, 0, 128, 0.6); transform: scale(1.02); }
        100% { box-shadow: 0 4px 15px rgba(255, 0, 128, 0.4); transform: scale(1); }
    }

    /* CONTAINERS & CARDS */
    .feature-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    .feature-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }

    /* RESULT BOXES */
    .prediction-box {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-out;
    }
    .stay-prediction {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .leave-prediction {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        color: #721c24;
    }

    /* CUSTOM PROGRESS BARS */
    .progress-bar-container {
        width: 100%;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 5px 0 15px 0;
        height: 25px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .progress-bar-green {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #34ce57);
        border-radius: 10px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .progress-bar-red {
        height: 100%;
        background: linear-gradient(90deg, #dc3545, #ff4d5e);
        border-radius: 10px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* PREDICT BUTTON STYLING */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #ff0080, #ff8c00, #40e0d0, #ff0080);
        background-size: 400% 400%;
        color: white !important;
        font-size: 1.4rem;
        font-weight: 800 !important;
        padding: 0.8rem 2.5rem;
        border-radius: 50px;
        border: none !important;
        outline: none !important;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(255, 0, 128, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: gradientShift 3s ease infinite;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 25px rgba(255, 0, 128, 0.6);
    }
    .stButton>button:active {
        transform: translateY(2px);
    }

    /* CUSTOM EXPANDER */
    div[data-testid="stExpander"] details summary {
        background-color: #1E3A5F !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1.1rem !important;
    }
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 8px 8px 0 0 !important;
    }
    div[data-testid="stExpander"] details > div {
        border: 1px solid #1E3A5F !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        background-color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
HF_REPO_ID = "Zlib2/RFC"
MODEL_FILENAME = "final_random_forest_model.joblib"

# Define the 5 selected features (in exact order expected by model)
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
# STATE MANAGEMENT (Sync Sliders & Inputs)
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
# MAIN APP LOGIC
# ============================================================================
def main():
    # Header Section
    st.markdown('<div class="main-header">👥 Employee Turnover AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced analytics to predict employee retention risks</div>', unsafe_allow_html=True)
    
    # Load Model
    model = load_model_from_huggingface()
    
    if model is None:
        st.warning("⚠️ Model could not be loaded. Please check your internet connection or repository status.")
        return

    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/groups.png", width=150)
        st.title("Model Insights")
        
        st.markdown("### 📊 Specifications")
        st.markdown("""
        <div class="metric-card">
            <b>Algorithm:</b> Random Forest<br>
            <b>Accuracy:</b> ~99% (Test)<br>
            <b>Features:</b> 5 Key Metrics
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Prediction Classes")
        st.info("**0: Stay** (Low Risk)")
        st.error("**1: Leave** (High Risk)")
        
        st.markdown("---")
        st.markdown(f"🔗 [Model Repository](https://huggingface.co/{HF_REPO_ID})")

    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    
    # Initialize session state if not present
    if 'satisfaction_level' not in st.session_state: st.session_state.satisfaction_level = 0.50
    if 'last_evaluation' not in st.session_state: st.session_state.last_evaluation = 0.70
    
    st.markdown("### 📝 Employee Metrics")
    
    # Container for inputs to look like a card
    with st.container():
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        
        # --- Row 1: Satisfaction & Evaluation ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### 😊 Satisfaction Level")
            sc1, sc2 = st.columns([3, 1])
            with sc1:
                st.slider("", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, 
                          key="sat_slider", on_change=sync_satisfaction_slider, label_visibility="collapsed")
            with sc2:
                st.number_input("", 0.0, 1.0, st.session_state.satisfaction_level, 0.01, 
                                key="sat_input", on_change=sync_satisfaction_input, label_visibility="collapsed")
        
        with c2:
            st.markdown("##### 📊 Last Evaluation")
            ec1, ec2 = st.columns([3, 1])
            with ec1:
                st.slider("", 0.0, 1.0, st.session_state.last_evaluation, 0.01, 
                          key="eval_slider", on_change=sync_evaluation_slider, label_visibility="collapsed")
            with ec2:
                st.number_input("", 0.0, 1.0, st.session_state.last_evaluation, 0.01, 
                                key="eval_input", on_change=sync_evaluation_input, label_visibility="collapsed")

        st.markdown("---")

        # --- Row 2: Projects, Years, Hours ---
        c3, c4, c5 = st.columns(3)
        
        with c3:
            number_project = st.number_input("📁 Projects", 2, 7, 4, 1, help="Number of ongoing projects")
        with c4:
            time_spend_company = st.number_input("📅 Years at Company", 2, 10, 3, 1, help="Total years employed")
        with c5:
            average_monthly_hours = st.number_input("⏰ Monthly Hours", 96, 310, 200, 5, help="Average working hours per month")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================================
    # PREDICTION LOGIC
    # ========================================================================
    
    # Spacing
    st.write("")
    st.write("")
    
    # Centered Button
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        predict_btn = st.button("🔮 ANALYZE TURNOVER RISK")

    if predict_btn:
        with st.spinner("Processing employee data..."):
            # Prepare Input
            input_data = pd.DataFrame([{
                'satisfaction_level': st.session_state.satisfaction_level,
                'time_spend_company': time_spend_company,
                'average_monthly_hours': average_monthly_hours,
                'number_project': number_project,
                'last_evaluation': st.session_state.last_evaluation
            }])[BEST_FEATURES]

            # Predict
            pred = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]
            prob_stay = probs[0] * 100
            prob_leave = probs[1] * 100

        st.markdown("---")
        
        # ====================================================================
        # RESULTS DISPLAY
        # ====================================================================
        r1, r2 = st.columns([1, 1.5])
        
        with r1:
            st.markdown("### 🚦 Verdict")
            if pred == 0:
                st.markdown(f"""
                <div class="prediction-box stay-prediction">
                    <h1 style="margin:0; font-size: 3rem;">STAY</h1>
                    <p style="margin-top:10px; font-weight:bold;">Likely to Retain</p>
                    <p style="font-size:0.9rem;">The model predicts this employee is happy and engaged.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box leave-prediction">
                    <h1 style="margin:0; font-size: 3rem;">LEAVE</h1>
                    <p style="margin-top:10px; font-weight:bold;">High Turnover Risk</p>
                    <p style="font-size:0.9rem;">Immediate intervention or retention strategy recommended.</p>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            st.markdown("### 📈 Risk Probability")
            
            # Stay Bar
            st.write(f"**Confidence to STAY:** {prob_stay:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-green" style="width: {prob_stay}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leave Bar
            st.write(f"**Confidence to LEAVE:** {prob_leave:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-red" style="width: {prob_leave}%;"></div>
            </div>
            """, unsafe_allow_html=True)

        # ====================================================================
        # SUMMARY EXPANDER
        # ====================================================================
        st.write("")
        c_exp1, c_exp2, c_exp3 = st.columns([1, 4, 1])
        with c_exp2:
            with st.expander("📋 View Input Data Summary"):
                summary_df = pd.DataFrame({
                    'Metric': ['Satisfaction', 'Evaluation', 'Projects', 'Years', 'Hours'],
                    'Value': [
                        f"{st.session_state.satisfaction_level:.2f}",
                        f"{st.session_state.last_evaluation:.2f}",
                        number_project,
                        time_spend_company,
                        average_monthly_hours
                    ]
                })
                st.table(summary_df)

if __name__ == "__main__":
    main()
