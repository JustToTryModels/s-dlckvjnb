import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
import time

# ============================================================================
# PAGE CONFIG & GOOGLE FONTS
# ============================================================================
st.set_page_config(
    page_title="TurnoverGuard • Employee Retention Prediction",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Professional fonts + subtle animations
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {font-family: 'Inter', sans-serif;}
    .big-title {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0A66C2, #14ae5c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.25rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .css-1d391kg {padding-top: 2rem;} /* reduce top padding */
    
    /* Sidebar styling */
    .css-1lcbmhc {background-color: #0f1b2e !important;}
    .sidebar-title {color: #14ae5c; font-weight: 600; font-size: 1.4rem;}
    .sidebar-text {color: #a8b5c8; font-size: 0.95rem;}
    
    /* Input cards */
    .input-card {
        background: white;
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid #eef2f6;
    }
    .input-card:hover {transform: translateY(-4px); box-shadow: 0 12px 35px rgba(0,0,0,0.12);}
    
    /* Modern sliders */
    .stSlider > div > div > div > div {background: #0A66C2;}
    .stSlider .thumb {background: #0A66C2; box-shadow: 0 0 15px rgba(10,102,194,0.4);}
    
    /* Predict button - elegant & premium */
    .predict-btn {
        background: linear-gradient(135deg, #0A66C2 0%, #14ae5c 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 8px 25px rgba(10,102,194,0.3);
        transition: all 0.4s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .predict-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(10,102,194,0.45);
    }
    .predict-btn:active {
        transform: translateY(1px);
    }
    
    /* Result cards */
    .result-card {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        animation: fadeInUp 0.8s ease;
        box-shadow: 0 15px 40px rgba(0,0,0,0.1);
    }
    .stay-card {
        background: linear-gradient(135deg, #d4f4dd 0%, #e8f5e9 100%);
        border: 2px solid #14ae5c;
    }
    .leave-card {
        background: linear-gradient(135deg, #ffe0e0 0%, #ffebeb 100%);
        border: 2px solid #e63946;
    }
    .prob-bar {
        height: 14px;
        border-radius: 7px;
        background: #e0e0e0;
        overflow: hidden;
        margin: 12px 0;
    }
    .prob-fill-stay {
        height: 100%;
        background: #14ae5c;
        border-radius: 7px;
        transition: width 1.2s ease-out;
    }
    .prob-fill-leave {
        height: 100%;
        background: #e63946;
        border-radius: 7px;
        transition: width 1.2s ease-out;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(30px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .fade-in {animation: fadeInUp 0.8s ease;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .big-title {font-size: 2.5rem;}
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIG
# ============================================================================
HF_REPO_ID = "Zlib2/RFC"
MODEL_FILENAME = "final_random_forest_model.joblib"
BEST_FEATURES = [
    "satisfaction_level", "time_spend_company", "average_monthly_hours",
    "number_project", "last_evaluation"
]

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        return joblib.load(path)
    except Exception as e:
        st.error("Failed to load model from Hugging Face.")
        return None

model = load_model()
if model is None:
    st.stop()

# ============================================================================
# SIDEBAR - Premium look
# ============================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-title">🛡️ TurnoverGuard Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-text">State-of-the-art Random Forest model<br>trained on 15,000+ employee records</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Selected Features**")
    icons = ["😊", "📊", "📅", "📁", "⏰"]
    for icon, feat in zip(icons, BEST_FEATURES):
        name = feat.replace("_", " ").title()
        st.markdown(f"{icon} {name}")
    st.markdown("---")
    st.markdown("**Performance**")
    st.markdown("• Accuracy: 99.1%  \n• AUC: 0.998  \n• Balanced classes")
    st.markdown("---")
    st.markdown("[View Model on Hugging Face](https://huggingface.co/Zlib2/RFC)")

# ============================================================================
# MAIN UI
# ============================================================================
st.markdown('<h1 class="big-title">Employee Retention Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict turnover risk with 99%+ accuracy using only 5 key indicators</p>', unsafe_allow_html=True)

# Input cards in grid
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("#### 😊 Satisfaction Level")
    satisfaction_level = st.slider(
        "", min_value=0.0, max_value=1.0, value=0.50, step=0.01,
        help="How satisfied is the employee? (0 = very dissatisfied, 1 = very satisfied)",
        key="sat"
    )
    st.markdown(f"<p style='text-align:center; color:#0A66C2; font-size:1.1rem; font-weight:600;'>{satisfaction_level:.2f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='input-card' style='margin-top:1.5rem;'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Last Evaluation Score")
    last_evaluation = st.slider(
        "", min_value=0.0, max_value=1.0, value=0.70, step=0.01,
        help="Latest performance evaluation (0 = poor, 1 = outstanding)",
        key="eval"
    )
    st.markdown(f"<p style='text-align:center; color:#0A66C2; font-size:1.1rem; font-weight:600;'>{last_evaluation:.2f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("#### 📅 Years at Company")
    time_spend_company = st.number_input(
        "", min_value=1, max_value=40, value=3, step=1,
        help="How long has the employee been with the company?",
        key="years"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='input-card' style='margin-top:1.5rem;'>", unsafe_allow_html=True)
    st.markdown("#### 📁 Number of Projects")
    number_project = st.number_input(
        "", min_value=1, max_value=10, value=4, step=1,
        help="How many projects is the employee currently assigned to?",
        key="projects"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='input-card' style='margin-top:1.5rem;'>", unsafe_allow_html=True)
    st.markdown("#### ⏰ Average Monthly Hours")
    average_monthly_hours = st.number_input(
        "", min_value=80, max_value=350, value=200, step=5,
        help="Average hours worked per month (last 12 months)",
        key="hours"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Predict button
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict Retention Risk", key="predict", use_container_width=True, 
                           help="Click to run prediction", 
                           type="primary",
                           on_click=lambda: st.session_state.update(predicted=False))

if predict_clicked or st.session_state.get("predicted", False):
    with st.spinner("Analyzing employee profile..."):
        time.sleep(1.2)  # small delay for premium feel
        
        input_df = pd.DataFrame([{
            "satisfaction_level": satisfaction_level,
            "time_spend_company": time_spend_company,
            "average_monthly_hours": average_monthly_hours,
            "number_project": number_project,
            "last_evaluation": last_evaluation
        }])[BEST_FEATURES]
        
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        prob_stay = prob[0] * 100
        prob_leave = prob[1] * 100
        
        st.session_state.predicted = True
        
        # Results
        col1, col2 = st.columns([1.8, 1])
        
        with col1:
            if pred == 0:
                st.markdown(f"""
                <div class="result-card stay-card fade-in">
                    <h1 style="font-size:4rem; margin:0;">✅</h1>
                    <h2 style="color:#14ae5c; margin:1rem 0 0.5rem;">Likely to STAY</h2>
                    <p style="font-size:1.2rem; color:#2d6a4f;">This employee shows strong retention signals</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card leave-card fade-in">
                    <h1 style="font-size:4rem; margin:0;">⚠️</h1>
                    <h2 style="color:#e63946; margin:1rem 0 0.5rem;">At Risk of Leaving</h2>
                    <p style="font-size:1.2rem; color:#9b2226;">High turnover probability – consider retention actions</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='fade-in' style='margin-top:2rem;'>", unsafe_allow_html=True)
            st.markdown("#### Confidence Levels")
            st.markdown(f"<div class='prob-bar'><div class='prob-fill-stay' style='width:{prob_stay}%'></div></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; color:#14ae5c; font-weight:600;'>Stay → {prob_stay:.1f}%</p>", unsafe_allow_html=True)
            
            st.markdown(f"<div class='prob-bar'><div class='prob-fill-leave' style='width:{prob_leave}%'></div></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; color:#e63946; font-weight:600;'>Leave → {prob_leave:.1f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Input summary (elegant expander)
        with st.expander("📋 View Input Summary", expanded=False):
            summary = pd.DataFrame({
                "Feature": ["Satisfaction Level", "Last Evaluation", "Years at Company", "Number of Projects", "Avg Monthly Hours"],
                "Value": [f"{satisfaction_level:.2f}", f"{last_evaluation:.2f}", f"{time_spend_company} years", f"{number_project}", f"{average_monthly_hours} hrs"]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("<br><br>", unsafe_allow_html=True)
