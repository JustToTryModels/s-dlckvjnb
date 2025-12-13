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
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #1E3A5F;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1E3A5F 0%, #00B4D8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out 0.3s both;
    }
    
    /* Animations */
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
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(0, 180, 216, 0.4);
        }
        50% {
            transform: scale(1.02);
            box-shadow: 0 0 0 10px rgba(0, 180, 216, 0);
        }
    }
    
    /* Prediction Box */
    .prediction-box {
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        animation: slideIn 0.6s ease-out;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    .stay-prediction {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #06D6A0;
    }
    
    .leave-prediction {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #EF476F;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #00B4D8;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateX(5px);
    }
    
    /* Feature Box */
    .feature-box {
        background: linear-gradient(135deg, #e7f3ff 0%, #d1e7fd 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid #b8daff;
        animation: fadeInUp 0.6s ease-out 0.5s both;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00B4D8 0%, #1E3A5F 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.4);
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 180, 216, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(1px) scale(0.98);
    }
    
    .stButton > button::before {
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
        transition: left 0.7s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Loading Spinner */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #00B4D8;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Progress Bars */
    .progress-bar-container {
        width: 100%;
        background-color: #e9ecef;
        border-radius: 15px;
        margin: 10px 0 20px 0;
        height: 30px;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 15px;
        transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .progress-bar-green {
        background: linear-gradient(90deg, #06D6A0 0%, #118AB2 100%);
    }
    
    .progress-bar-red {
        background: linear-gradient(90deg, #EF476F 0%, #FF9A76 100%);
    }
    
    /* Sliders */
    .stSlider {
        padding: 10px 0;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00B4D8 0%, #1E3A5F 100%);
        height: 6px;
        border-radius: 3px;
    }
    
    .stSlider > div > div > div > div > div {
        background: white;
        border: 2px solid #00B4D8;
        box-shadow: 0 2px 8px rgba(0, 180, 216, 0.3);
    }
    
    /* Number Inputs */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #ced4da;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
        font-weight: 500;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #00B4D8;
        box-shadow: 0 0 0 0.2rem rgba(0, 180, 216, 0.25);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        color: white;
        border-radius: 8px;
        font-weight: 500;
        transition: background 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #2E5A8F 0%, #1E3A5F 100%);
    }
    
    .streamlit-expanderContent {
        border: 1px solid #1E3A5F;
        border-top: none;
        border-radius: 0 0 8px 8px;
        background: white;
    }
    
    /* DataFrame */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #1E3A5F 0%, #00B4D8 100%);
        color: white;
        font-weight: 600;
        padding: 12px;
    }
    
    .dataframe td {
        padding: 10px 12px;
        border-bottom: 1px solid #e9ecef;
    }
    
    .dataframe tr:hover {
        background-color: #f8f9fa;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .prediction-box {
            padding: 1.5rem;
        }
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
        with st.spinner("🤖 Loading AI Model..."):
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
    # Header with animation
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict employee retention risk with AI-powered insights</p>', unsafe_allow_html=True)
    
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
        st.markdown("### ℹ️ Model Information")
        
        with st.container():
            st.markdown("""
            <div class="metric-card">
                <strong>🤖 Algorithm:</strong> Random Forest Classifier<br><br>
                <strong>📊 Features:</strong> 5 key predictors<br><br>
                <strong>⚖️ Class Weight:</strong> Balanced<br><br>
                <strong>🎯 Accuracy:</strong> Optimized for HR analytics
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 Key Features")
        for i, feat in enumerate(BEST_FEATURES, 1):
            feat_display = feat.replace('_', ' ').title()
            st.write(f"{i}. **{feat_display}**")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Prediction Classes")
        st.info("""
        **0 → Stay**: Employee likely to remain  
        **1 → Leave**: Employee likely to depart
        """)
        
        st.markdown("---")
        
        st.markdown(f"[🔗 View Model on Hugging Face](https://huggingface.co/{HF_REPO_ID})")
    
    # ========================================================================
    # MAIN INPUT SECTION
    # ========================================================================
    st.markdown("---")
    st.subheader("📝 Enter Employee Information")
    
    st.markdown("""
    <div class="feature-box">
        <strong>📌 Instructions:</strong> Adjust the values below to see real-time prediction insights. 
        Each parameter has been selected based on its impact on employee retention.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # INPUT ROWS
    # ========================================================================
    # Row 1: Satisfaction & Evaluation
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.write("😊 **Satisfaction Level**")
        sat_col1, sat_col2 = st.columns([4, 1])
        with sat_col1:
            st.slider(
                "Satisfaction",
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
                "Satisfaction Value",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.satisfaction_level,
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key="sat_input",
                on_change=sync_satisfaction_input
            )
    
    with row1_col2:
        st.write("📊 **Last Evaluation Score**")
        eval_col1, eval_col2 = st.columns([4, 1])
        with eval_col1:
            st.slider(
                "Evaluation",
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
                "Evaluation Value",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.last_evaluation,
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key="eval_input",
                on_change=sync_evaluation_input
            )
    
    # Row 2: Years & Projects
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        time_spend_company = st.number_input(
            "📅 Years at Company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            help="Total years of service at the organization"
        )
    
    with row2_col2:
        number_project = st.number_input(
            "📁 Number of Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Current number of assigned projects"
        )
    
    # Row 3: Monthly Hours
    average_monthly_hours = st.number_input(
        "⏰ Average Monthly Hours",
        min_value=80,
        max_value=350,
        value=200,
        step=5,
        help="Average monthly working hours (standard: ~160-200 hrs)"
    )
    
    # Create input dictionary
    input_data = {
        'satisfaction_level': st.session_state.satisfaction_level,
        'time_spend_company': time_spend_company,
        'average_monthly_hours': average_monthly_hours,
        'number_project': number_project,
        'last_evaluation': st.session_state.last_evaluation
    }
    
    # ========================================================================
    # PREDICTION BUTTON
    # ========================================================================
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Generate Prediction", use_container_width=True)
    
    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================
    if predict_button:
        # Show loading animation
        with st.spinner('🤖 Analyzing employee data...'):
            time.sleep(1.5)  # Simulate processing time for better UX
            
            # Create input DataFrame
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
                    <div style="font-size: 4rem; margin-bottom: 1rem;">✅</div>
                    <h1 style="color: #06D6A0; margin: 0;">STAY</h1>
                    <p style="font-size: 1.2rem; margin-top: 1rem; color: #2B2D42;">
                        Employee is <strong>likely to remain</strong> with the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box leave-prediction">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
                    <h1 style="color: #EF476F; margin: 0;">LEAVE</h1>
                    <p style="font-size: 1.2rem; margin-top: 1rem; color: #2B2D42;">
                        Employee is <strong>at risk of leaving</strong> the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Confidence Levels")
            
            # Stay probability
            st.write(f"**Probability of Staying:** {prob_stay:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-fill progress-bar-green" style="width: {prob_stay}%;">
                    {prob_stay:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leave probability
            st.write(f"**Probability of Leaving:** {prob_leave:.1f}%")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-fill progress-bar-red" style="width: {prob_leave}%;">
                    {prob_leave:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk indicator
            risk_level = "🔴 High Risk" if prob_leave > 60 else "🟡 Medium Risk" if prob_leave > 30 else "🟢 Low Risk"
            st.markdown(f"""
            <div class="info-box" style="margin-top: 1.5rem;">
                <strong>Risk Assessment:</strong> {risk_level}
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
                
                # Style the dataframe
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Feature": st.column_config.TextColumn(width="medium"),
                        "Value": st.column_config.TextColumn(width="small")
                    }
                )
                
                # Add download button
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Summary",
                    data=csv,
                    file_name="employee_prediction_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
