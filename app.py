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
    .hf-badge {
        background-color: #ff9d00;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A5F;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2E5A8F;
        color: white;
    }
    .info-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
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
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">👥 Employee Turnover Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether an employee is likely to leave the company</p>', unsafe_allow_html=True)
    
    # Hugging Face badge
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f'<p style="text-align: center;"><span class="hf-badge">🤗 Powered by Hugging Face</span></p>',
            unsafe_allow_html=True
        )
    
    # Load model
    with st.spinner("🔄 Loading model from Hugging Face..."):
        model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    
    st.success("✅ Model loaded successfully!")
    
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
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction Level
        satisfaction_level = st.slider(
            "😊 Satisfaction Level",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Employee satisfaction level (0 = Very Dissatisfied, 1 = Very Satisfied)"
        )
        
        # Time Spent at Company
        time_spend_company = st.number_input(
            "📅 Years at Company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            help="Number of years the employee has worked at the company"
        )
        
        # Average Monthly Hours
        average_monthly_hours = st.number_input(
            "⏰ Average Monthly Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            help="Average number of hours worked per month"
        )
    
    with col2:
        # Number of Projects
        number_project = st.number_input(
            "📁 Number of Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of projects the employee is currently working on"
        )
        
        # Last Evaluation
        last_evaluation = st.slider(
            "📊 Last Evaluation Score",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            help="Last performance evaluation score (0 = Poor, 1 = Excellent)"
        )
    
    # ========================================================================
    # INPUT SUMMARY
    # ========================================================================
    st.markdown("---")
    st.subheader("📋 Input Summary")
    
    # Create input dictionary
    input_data = {
        'satisfaction_level': satisfaction_level,
        'time_spend_company': time_spend_company,
        'average_monthly_hours': average_monthly_hours,
        'number_project': number_project,
        'last_evaluation': last_evaluation
    }
    
    # Display as a nice table
    summary_df = pd.DataFrame({
        'Feature': [
            '😊 Satisfaction Level',
            '📅 Years at Company',
            '⏰ Average Monthly Hours',
            '📁 Number of Projects',
            '📊 Last Evaluation'
        ],
        'Value': [
            f"{satisfaction_level:.2f}",
            f"{time_spend_company} years",
            f"{average_monthly_hours} hours",
            f"{number_project} projects",
            f"{last_evaluation:.2f}"
        ]
    })
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
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
            
            # Stay probability
            st.write(f"**Probability of Staying:** {prob_stay:.1f}%")
            st.progress(prob_stay / 100)
            
            # Leave probability
            st.write(f"**Probability of Leaving:** {prob_leave:.1f}%")
            st.progress(prob_leave / 100)
        
        # ====================================================================
        # CONFIDENCE LEVEL
        # ====================================================================
        st.markdown("---")
        
        max_prob = max(prob_stay, prob_leave)
        
        if max_prob >= 80:
            confidence = "🟢 High Confidence"
            confidence_color = "#28a745"
            confidence_desc = "The model is very confident in this prediction."
        elif max_prob >= 60:
            confidence = "🟡 Moderate Confidence"
            confidence_color = "#ffc107"
            confidence_desc = "The model has moderate confidence. Consider additional factors."
        else:
            confidence = "🔴 Low Confidence"
            confidence_color = "#dc3545"
            confidence_desc = "The prediction is uncertain. More information may be needed."
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                        border-left: 5px solid {confidence_color}; text-align: center;">
                <h3 style="margin: 0;">{confidence}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #666;">{confidence_desc}</p>
                <p style="margin: 0.5rem 0 0 0;"><strong>Confidence Score: {max_prob:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # RECOMMENDATIONS (only if prediction is LEAVE)
        # ====================================================================
        if prediction == 1:
            st.markdown("---")
            st.subheader("💡 Retention Recommendations")
            
            # Personalized recommendations based on input values
            recommendations = []
            
            if satisfaction_level < 0.5:
                recommendations.append("📉 **Low Satisfaction Detected:** Schedule a one-on-one meeting to understand concerns and improve job satisfaction.")
            
            if average_monthly_hours > 250:
                recommendations.append("⏰ **High Working Hours:** Employee may be overworked. Consider redistributing workload or hiring additional support.")
            elif average_monthly_hours < 150:
                recommendations.append("⏰ **Low Working Hours:** Employee may feel underutilized. Consider assigning more meaningful projects.")
            
            if number_project > 5:
                recommendations.append("📁 **Too Many Projects:** Reduce project load to prevent burnout and improve focus.")
            elif number_project < 3:
                recommendations.append("📁 **Few Projects:** Employee may need more engagement. Consider involving them in new initiatives.")
            
            if time_spend_company >= 4 and last_evaluation > 0.7:
                recommendations.append("🎯 **Experienced & High Performer:** Consider promotion opportunities or leadership roles to retain this valuable employee.")
            
            if last_evaluation < 0.5:
                recommendations.append("📊 **Low Evaluation Score:** Provide training, mentorship, and clear performance improvement plans.")
            
            # Default recommendations
            recommendations.extend([
                "💰 **Review Compensation:** Ensure salary and benefits are competitive with market rates.",
                "🚀 **Career Development:** Discuss growth opportunities and create a clear career path.",
                "🏆 **Recognition:** Acknowledge contributions and achievements regularly."
            ])
            
            # Display recommendations
            for rec in recommendations[:6]:  # Show top 6 recommendations
                st.warning(rec)
        
        # ====================================================================
        # DETAILED INPUT BREAKDOWN
        # ====================================================================
        with st.expander("📋 View Detailed Input Data"):
            st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
