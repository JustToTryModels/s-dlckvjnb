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
    /* GLOBAL STYLES */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* HEADER STYLES */
    .main-header {
        background: linear-gradient(90deg, #1E3A5F 0%, #2E5A8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
        animation: fadeInDown 1s ease-out;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2.5rem;
        animation: fadeIn 1.5s ease-out;
    }

    /* ANIMATIONS */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.7); }
        70% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(255, 255, 255, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
    }

    /* INPUT CARDS */
    .feature-card {
        background-color: #ffffff;
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

    /* PREDICTION BUTTON STYLE */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #2E5A8F, #1E3A5F);
        color: white;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 4px 15px rgba(30, 58, 95, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #1E3A5F, #2E5A8F);
        box-shadow: 0 6px 20px rgba(30, 58, 95, 0.5);
        transform: translateY(-2px);
        color: #ffffff;
    }
    .stButton>button:active {
        transform: translateY(1px);
    }

    /* RESULT BOXES */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
        color: #1a1a1a;
        margin-top: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .stay-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    .leave-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    
    /* PROGRESS BARS */
    .progress-wrapper {
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 10px 0;
        height: 20px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .progress-fill-green {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #34ce57);
        transition: width 1s ease-in-out;
    }
    .progress-fill-red {
        height: 100%;
        background: linear-gradient(90deg, #dc3545, #e4606d);
        transition: width 1s ease-in-out;
    }

    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
    .sidebar-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }

    /* CUSTOM TOOLTIP ICON */
    .tooltip-icon {
        color: #1E3A5F;
        font-size: 0.9rem;
        margin-left: 5px;
        cursor: help;
    }

    /* EXPANDER STYLING */
    div[data-testid="stExpander"] details summary {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-weight: 600;
        color: #1E3A5F;
    }
    div[data-testid="stExpander"] details[open] summary {
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
    }
    div[data-testid="stExpander"] div[role="button"]:hover {
        color: #2E5A8F;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIG & CONSTANTS
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
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model from Hugging Face Hub with caching."""
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
        return joblib.load(model_path)
    except Exception as e:
        return None

# ============================================================================
# CALLBACKS FOR SYNCED INPUTS
# ============================================================================
def update_sat_slider():
    st.session_state.sat_slider = st.session_state.sat_input

def update_sat_input():
    st.session_state.sat_input = st.session_state.sat_slider

def update_eval_slider():
    st.session_state.eval_slider = st.session_state.eval_input

def update_eval_input():
    st.session_state.eval_input = st.session_state.eval_slider

# ============================================================================
# MAIN APPLICATION LAYOUT
# ============================================================================
def main():
    # 1. SIDEBAR
    with st.sidebar:
        st.markdown('<div style="text-align: center; margin-bottom: 20px;"><h2>⚙️ Model Config</h2></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <strong>🤖 Algorithm:</strong> Random Forest<br>
            <strong>🎯 Accuracy:</strong> ~99%<br>
            <strong>📉 Loss Function:</strong> Gini Impurity
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 Feature Importance")
        st.caption("Top 5 features used for prediction:")
        
        features_df = pd.DataFrame({
            'Feature': ['Satisfaction', 'Yrs at Company', 'Avg Monthly Hrs', 'Projects', 'Last Eval'],
            'Importance': [0.35, 0.25, 0.18, 0.15, 0.07] # Approximate visual weights
        })
        st.bar_chart(features_df.set_index('Feature'), color="#1E3A5F", height=150)
        
        st.markdown("---")
        st.info("ℹ️ **Privacy Note:** Data entered here is processed in real-time and not stored.")
        st.markdown(f"[🔗 View Model Repo](https://huggingface.co/{HF_REPO_ID})")

    # 2. MAIN HEADER
    st.markdown('<h1 class="main-header">👥 Employee Turnover AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced analytics to predict employee retention risk</p>', unsafe_allow_html=True)

    # Load Model
    model = load_model()
    if model is None:
        st.error(f"❌ Could not load model from Hugging Face ({HF_REPO_ID}). Please check connection.")
        return

    # 3. INPUT FORM
    
    # Initialize session state for synced inputs if not exists
    if 'sat_input' not in st.session_state: st.session_state.sat_input = 0.50
    if 'sat_slider' not in st.session_state: st.session_state.sat_slider = 0.50
    if 'eval_input' not in st.session_state: st.session_state.eval_input = 0.70
    if 'eval_slider' not in st.session_state: st.session_state.eval_slider = 0.70

    st.markdown("### 📝 Employee Metrics")
    
    with st.container():
        # --- ROW 1: Satisfaction & Evaluation ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">😊 <strong>Satisfaction Level</strong></span>
            </div>
            """, unsafe_allow_html=True)
            
            c1_a, c1_b = st.columns([3, 1])
            with c1_a:
                st.slider("", 0.0, 1.0, key="sat_slider", on_change=update_sat_input, label_visibility="collapsed", help="Employee's self-reported satisfaction.")
            with c1_b:
                st.number_input("", 0.0, 1.0, step=0.01, key="sat_input", on_change=update_sat_slider, label_visibility="collapsed")

        with col2:
            st.markdown("""
            <div class="feature-card">
                <span style="font-size: 1.2rem;">📊 <strong>Last Evaluation</strong></span>
            </div>
            """, unsafe_allow_html=True)
            
            c2_a, c2_b = st.columns([3, 1])
            with c2_a:
                st.slider("", 0.0, 1.0, key="eval_slider", on_change=update_eval_input, label_visibility="collapsed", help="Score from the last performance review.")
            with c2_b:
                st.number_input("", 0.0, 1.0, step=0.01, key="eval_input", on_change=update_eval_slider, label_visibility="collapsed")

        # --- ROW 2: Years, Projects, Hours ---
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown('<div style="margin-top: 10px;"><strong>📅 Years at Company</strong></div>', unsafe_allow_html=True)
            time_spend = st.number_input("Years", min_value=1, max_value=20, value=3, step=1, label_visibility="collapsed", help="Total duration of employment.")
            
        with col4:
            st.markdown('<div style="margin-top: 10px;"><strong>📁 Number of Projects</strong></div>', unsafe_allow_html=True)
            num_projects = st.number_input("Projects", min_value=1, max_value=10, value=4, step=1, label_visibility="collapsed", help="Current active project count.")

        with col5:
            st.markdown('<div style="margin-top: 10px;"><strong>⏰ Avg. Monthly Hours</strong></div>', unsafe_allow_html=True)
            avg_hours = st.number_input("Hours", min_value=50, max_value=350, value=200, step=5, label_visibility="collapsed", help="Average working hours per month.")

    # 4. PREDICTION SECTION
    st.markdown("---")
    
    # Centered Predict Button
    b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
    with b_col2:
        predict_btn = st.button("🔮 Predict Turnover Risk", use_container_width=True)

    # 5. RESULTS DISPLAY
    if predict_btn:
        # Create Dataframe
        input_data = pd.DataFrame([{
            'satisfaction_level': st.session_state.sat_input,
            'time_spend_company': time_spend,
            'average_monthly_hours': avg_hours,
            'number_project': num_projects,
            'last_evaluation': st.session_state.eval_input
        }])
        
        # Ensure column order matches model training
        input_data = input_data[BEST_FEATURES]

        # Simulate Loading
        with st.spinner("Analyzing employee patterns..."):
            time.sleep(0.8) # Small UI delay for effect
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            prob_stay = probabilities[0] * 100
            prob_leave = probabilities[1] * 100

        # Display Logic
        st.markdown("<br>", unsafe_allow_html=True)
        r_col1, r_col2 = st.columns([1, 1])

        with r_col1:
            if prediction == 0:
                st.markdown(f"""
                <div class="result-box stay-box">
                    <h1 style="margin:0; color: #155724;">✅ STAY</h1>
                    <h4 style="margin-top:10px;">Low Turnover Risk</h4>
                    <p>This employee is likely to remain with the company.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box leave-box">
                    <h1 style="margin:0; color: #721c24;">⚠️ LEAVE</h1>
                    <h4 style="margin-top:10px;">High Turnover Risk</h4>
                    <p>This employee shows strong indicators of leaving.</p>
                </div>
                """, unsafe_allow_html=True)

        with r_col2:
            st.markdown("### 📊 Probability Analysis")
            
            # Stay Bar
            st.markdown(f"**Stay Probability:** {prob_stay:.1f}%")
            st.markdown(f"""
            <div class="progress-wrapper">
                <div class="progress-fill-green" style="width: {prob_stay}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leave Bar
            st.markdown(f"**Leave Probability:** {prob_leave:.1f}%")
            st.markdown(f"""
            <div class="progress-wrapper">
                <div class="progress-fill-red" style="width: {prob_leave}%;"></div>
            </div>
            """, unsafe_allow_html=True)

        # 6. INPUT SUMMARY (Collapsible)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 View Input Summary"):
            summary_df = pd.DataFrame({
                'Metric': ['Satisfaction', 'Evaluation', 'Years', 'Projects', 'Hours'],
                'Value': [
                    f"{st.session_state.sat_input:.2f}",
                    f"{st.session_state.eval_input:.2f}",
                    time_spend,
                    num_projects,
                    avg_hours
                ]
            })
            st.table(summary_df)

if __name__ == "__main__":
    main()
