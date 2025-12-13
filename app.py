import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from huggingface_hub import hf_hub_download

# ============================================================================
# 1. PAGE CONFIGURATION & SESSION STATE
# ============================================================================
st.set_page_config(
    page_title="Employee Retention AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for synchronized inputs
if 'satisfaction_level' not in st.session_state:
    st.session_state.satisfaction_level = 0.50
if 'last_evaluation' not in st.session_state:
    st.session_state.last_evaluation = 0.70

# ============================================================================
# 2. ADVANCED CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* GLOBAL STYLES */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* HEADER STYLES */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1E3A5F, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
        animation: fadeIn 1.2s ease-out;
    }

    /* CARD CONTAINERS */
    .feature-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #f0f2f6;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    /* PREDICTION BUTTON */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563EB 0%, #1E3A5F 100%);
        color: white;
        font-weight: 700;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1E3A5F 0%, #2563EB 100%);
        box-shadow: 0 10px 15px rgba(37, 99, 235, 0.4);
        transform: scale(1.02);
    }
    .stButton > button:active {
        transform: scale(0.98);
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-metric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* RESULT BOXES */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1rem;
        animation: zoomIn 0.5s ease-out;
    }
    .result-safe {
        background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);
        border: 2px solid #22c55e;
        color: #15803d;
    }
    .result-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fef2f2 100%);
        border: 2px solid #ef4444;
        color: #b91c1c;
    }

    /* PROGRESS BARS */
    .custom-progress-bg {
        width: 100%;
        background-color: #e2e8f0;
        border-radius: 10px;
        height: 20px;
        margin-top: 5px;
        overflow: hidden;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }

    /* KEYFRAME ANIMATIONS */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translate3d(0, -20px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes zoomIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* TOOLTIP HELPER */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 3. HELPER FUNCTIONS & MODEL LOADING
# ============================================================================

# Configuration
HF_REPO_ID = "Zlib2/RFC"
MODEL_FILENAME = "final_random_forest_model.joblib"
BEST_FEATURES = [
    "satisfaction_level",
    "time_spend_company", 
    "average_monthly_hours",
    "number_project",
    "last_evaluation"
]

@st.cache_resource(show_spinner="Downloading Model...")
def load_model():
    """Load the model from Hugging Face with caching."""
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

# Sync functions for Sliders/Number Inputs
def sync_satisfaction_slider():
    st.session_state.satisfaction_level = st.session_state.sat_slider

def sync_satisfaction_input():
    st.session_state.satisfaction_level = st.session_state.sat_input

def sync_evaluation_slider():
    st.session_state.last_evaluation = st.session_state.eval_slider

def sync_evaluation_input():
    st.session_state.last_evaluation = st.session_state.eval_input

# ============================================================================
# 4. MAIN LAYOUT
# ============================================================================

def main():
    # --- Header Section ---
    st.markdown('<div class="main-header">🏢 Employee Retention AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced analytics to predict employee turnover risk</div>', unsafe_allow_html=True)

    # Load Model
    model = load_model()
    if not model:
        return

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/company.png", use_container_width=True) # Placeholder professional icon
        st.markdown("### ⚙️ Model Insights")
        
        st.markdown("""
        <div class="sidebar-metric">
            <strong>🤖 Algorithm</strong><br>
            <span style="color: #666;">Random Forest Classifier</span>
        </div>
        <div class="sidebar-metric">
            <strong>📊 Accuracy</strong><br>
            <span style="color: #666;">~98.5% (Test Set)</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("**Note:** This tool provides a probabilistic assessment based on historical data patterns.")
        st.markdown("---")
        st.markdown(f"[🔗 View Model Repository](https://huggingface.co/{HF_REPO_ID})")

    # --- Main Input Form ---
    st.markdown("### 📝 Employee Profile")
    
    with st.container():
        # -- Row 1: Satisfaction & Evaluation --
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("##### 😊 Satisfaction Level")
            st.caption("Self-reported satisfaction (0.00 - 1.00)")
            
            c1_sub, c2_sub = st.columns([3, 1])
            with c1_sub:
                st.slider("", 0.0, 1.0, key="sat_slider", 
                         value=st.session_state.satisfaction_level, 
                         on_change=sync_satisfaction_slider, label_visibility="collapsed")
            with c2_sub:
                st.number_input("", 0.0, 1.0, 0.01, key="sat_input", 
                               value=st.session_state.satisfaction_level, 
                               on_change=sync_satisfaction_input, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("##### 📈 Last Evaluation")
            st.caption("Performance score from last review (0.00 - 1.00)")
            
            c1_sub, c2_sub = st.columns([3, 1])
            with c1_sub:
                st.slider("", 0.0, 1.0, key="eval_slider", 
                         value=st.session_state.last_evaluation, 
                         on_change=sync_evaluation_slider, label_visibility="collapsed")
            with c2_sub:
                st.number_input("", 0.0, 1.0, 0.01, key="eval_input", 
                               value=st.session_state.last_evaluation, 
                               on_change=sync_evaluation_input, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        # -- Row 2: Projects, Hours, Tenure --
        col3, col4, col5 = st.columns(3)

        with col3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("##### 📂 Projects")
            number_project = st.number_input("Count", min_value=1, max_value=20, value=4, step=1, label_visibility="collapsed")
            st.caption("Active projects assigned")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("##### ⏱️ Monthly Hours")
            average_monthly_hours = st.number_input("Hours", min_value=50, max_value=400, value=200, step=5, label_visibility="collapsed")
            st.caption("Avg. working hours/month")
            st.markdown('</div>', unsafe_allow_html=True)

        with col5:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("##### 📅 Tenure")
            time_spend_company = st.number_input("Years", min_value=1, max_value=50, value=3, step=1, label_visibility="collapsed")
            st.caption("Years at the company")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Prediction Action ---
    st.write("") # Spacer
    
    # Centered Button
    b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
    with b_col2:
        predict_btn = st.button("✨ Analyze Retention Risk")

    # --- Logic & Results ---
    if predict_btn:
        with st.spinner("Processing employee data..."):
            time.sleep(0.8) # Artificial delay for UX "thinking" feel
            
            # Prepare Data
            input_data = pd.DataFrame([{
                'satisfaction_level': st.session_state.satisfaction_level,
                'last_evaluation': st.session_state.last_evaluation,
                'number_project': number_project,
                'average_monthly_hours': average_monthly_hours,
                'time_spend_company': time_spend_company
            }])[BEST_FEATURES]

            # Predict
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            prob_stay = probability[0] * 100
            prob_leave = probability[1] * 100

        # --- Display Results ---
        st.markdown("---")
        
        r_col1, r_col2 = st.columns([1, 1], gap="large")

        # Visual Status Box
        with r_col1:
            if prediction == 0:
                st.markdown(f"""
                <div class="result-box result-safe">
                    <h1 style="margin:0;">✅ STAY</h1>
                    <p style="font-size:1.1rem; margin-top:5px;">Low turnover risk detected.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-danger">
                    <h1 style="margin:0;">⚠️ LEAVE</h1>
                    <p style="font-size:1.1rem; margin-top:5px;">High turnover risk detected.</p>
                </div>
                """, unsafe_allow_html=True)

        # Probability Bars
        with r_col2:
            st.markdown("#### 📊 Risk Assessment Analysis")
            
            # Stay Bar
            st.write(f"**Likelihood to Stay:** {prob_stay:.1f}%")
            st.markdown(f"""
            <div class="custom-progress-bg">
                <div class="custom-progress-fill" style="width: {prob_stay}%; background-color: #22c55e;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer

            # Leave Bar
            st.write(f"**Likelihood to Leave:** {prob_leave:.1f}%")
            st.markdown(f"""
            <div class="custom-progress-bg">
                <div class="custom-progress-fill" style="width: {prob_leave}%; background-color: #ef4444;"></div>
            </div>
            """, unsafe_allow_html=True)

        # --- Collapsible Summary ---
        st.write("")
        st.write("")
        with st.expander("📋 View Input Data Summary", expanded=False):
            summary_df = pd.DataFrame({
                'Metric': ['Satisfaction', 'Evaluation', 'Projects', 'Monthly Hours', 'Tenure'],
                'Value': [
                    f"{st.session_state.satisfaction_level:.2f}",
                    f"{st.session_state.last_evaluation:.2f}",
                    str(number_project),
                    str(average_monthly_hours),
                    f"{time_spend_company} Years"
                ]
            })
            st.table(summary_df)

if __name__ == "__main__":
    main()
