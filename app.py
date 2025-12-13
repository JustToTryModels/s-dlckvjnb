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
# PROFESSIONAL CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* ===== GOOGLE FONTS IMPORT ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    /* ===== CSS VARIABLES FOR THEMING ===== */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --danger-gradient: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        --dark-bg: #0f0f1a;
        --card-bg: rgba(255, 255, 255, 0.03);
        --glass-bg: rgba(255, 255, 255, 0.08);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --text-muted: rgba(255, 255, 255, 0.5);
        --border-color: rgba(255, 255, 255, 0.1);
        --accent-blue: #667eea;
        --accent-purple: #764ba2;
        --accent-pink: #f093fb;
        --shadow-glow: 0 0 40px rgba(102, 126, 234, 0.3);
    }
    
    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== ANIMATED BACKGROUND PARTICLES ===== */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(118, 75, 162, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(240, 147, 251, 0.05) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* ===== HERO HEADER SECTION ===== */
    .hero-container {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            from 0deg,
            transparent 0deg 340deg,
            rgba(102, 126, 234, 0.1) 340deg 360deg
        );
        animation: heroRotate 20s linear infinite;
    }
    
    @keyframes heroRotate {
        100% { transform: rotate(360deg); }
    }
    
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: titleGlow 3s ease-in-out infinite alternate;
        position: relative;
        z-index: 1;
    }
    
    @keyframes titleGlow {
        0% { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5)); }
        100% { filter: drop-shadow(0 0 40px rgba(240, 147, 251, 0.5)); }
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }
    
    .sub-header-highlight {
        color: #f093fb;
        font-weight: 600;
    }
    
    /* ===== FLOATING BADGE ===== */
    .floating-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.85rem;
        color: #a8b5ff;
        margin-bottom: 1.5rem;
        animation: badgeFloat 3s ease-in-out infinite;
    }
    
    @keyframes badgeFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    .badge-dot {
        width: 8px;
        height: 8px;
        background: #38ef7d;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* ===== GLASSMORPHISM CARDS ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.05),
            transparent
        );
        transition: left 0.7s ease;
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.3),
            0 0 60px rgba(102, 126, 234, 0.1);
    }
    
    /* ===== CARD HEADER ===== */
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .card-icon {
        width: 50px;
        height: 50px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        background: var(--primary-gradient);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .card-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }
    
    .card-subtitle {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.5);
        margin: 0;
    }
    
    /* ===== INPUT FIELD STYLING ===== */
    .input-group {
        margin-bottom: 1.5rem;
    }
    
    .input-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.95rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 10px;
    }
    
    .input-label-icon {
        font-size: 1.2rem;
    }
    
    .input-hint {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.4);
        margin-top: 6px;
    }
    
    /* Streamlit Input Overrides */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }
    
    .stSlider > div > div > div > div {
        background: #ffffff !important;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.5) !important;
    }
    
    div[data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2) !important;
    }
    
    div[data-baseweb="input"] input {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Number Input Styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* ===== MEGA PREDICT BUTTON ===== */
    .stButton > button {
        width: 100%;
        background: linear-gradient(
            135deg,
            #667eea 0%,
            #764ba2 25%,
            #f093fb 50%,
            #764ba2 75%,
            #667eea 100%
        );
        background-size: 300% 300%;
        color: white !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.3rem;
        font-weight: 700 !important;
        padding: 1.2rem 3rem;
        border-radius: 16px;
        border: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 10px 40px rgba(102, 126, 234, 0.4),
            0 0 60px rgba(118, 75, 162, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: gradientFlow 4s ease infinite;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 
            0 20px 60px rgba(102, 126, 234, 0.5),
            0 0 80px rgba(240, 147, 251, 0.3);
        animation: gradientFlow 2s ease infinite;
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            120deg,
            transparent,
            rgba(255, 255, 255, 0.3),
            transparent
        );
        transition: left 0.6s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Remove focus outline */
    .stButton > button:focus,
    .stButton > button:focus-visible {
        outline: none !important;
        border: none !important;
    }
    
    /* ===== RESULT CARDS ===== */
    .result-container {
        display: flex;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .prediction-card {
        flex: 1;
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: resultSlideIn 0.6s ease-out;
    }
    
    @keyframes resultSlideIn {
        0% { 
            opacity: 0; 
            transform: translateY(30px) scale(0.95);
        }
        100% { 
            opacity: 1; 
            transform: translateY(0) scale(1);
        }
    }
    
    .stay-card {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.2) 0%, rgba(56, 239, 125, 0.1) 100%);
        border: 2px solid rgba(56, 239, 125, 0.3);
        box-shadow: 
            0 20px 40px rgba(17, 153, 142, 0.2),
            inset 0 0 60px rgba(56, 239, 125, 0.05);
    }
    
    .leave-card {
        background: linear-gradient(135deg, rgba(235, 51, 73, 0.2) 0%, rgba(244, 92, 67, 0.1) 100%);
        border: 2px solid rgba(244, 92, 67, 0.3);
        box-shadow: 
            0 20px 40px rgba(235, 51, 73, 0.2),
            inset 0 0 60px rgba(244, 92, 67, 0.05);
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: iconBounce 2s ease-in-out infinite;
    }
    
    @keyframes iconBounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .prediction-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .stay-card .prediction-title {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .leave-card .prediction-title {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .prediction-message {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
        line-height: 1.6;
    }
    
    /* ===== PROBABILITY SECTION ===== */
    .probability-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        animation: resultSlideIn 0.6s ease-out 0.2s both;
    }
    
    .probability-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .prob-item {
        margin-bottom: 1.5rem;
    }
    
    .prob-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .prob-text {
        font-size: 1rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .prob-value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    .prob-value-stay {
        color: #38ef7d;
    }
    
    .prob-value-leave {
        color: #f45c43;
    }
    
    /* Progress Bar Styles */
    .progress-container {
        width: 100%;
        height: 12px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 20px;
        position: relative;
        transition: width 1s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .progress-stay {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        box-shadow: 0 0 20px rgba(56, 239, 125, 0.4);
    }
    
    .progress-leave {
        background: linear-gradient(90deg, #eb3349, #f45c43);
        box-shadow: 0 0 20px rgba(244, 92, 67, 0.4);
    }
    
    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255, 255, 255, 0.3) 50%,
            transparent 100%
        );
        animation: progressShine 2s ease-in-out infinite;
    }
    
    @keyframes progressShine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* ===== SIDEBAR STYLING ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    .sidebar-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .sidebar-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .sidebar-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .sidebar-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .sidebar-card-value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
    }
    
    .feature-item:last-child {
        border-bottom: none;
    }
    
    .feature-item:hover {
        color: #ffffff;
        padding-left: 10px;
    }
    
    .feature-number {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
    }
    
    /* ===== EXPANDER STYLING ===== */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        overflow: hidden;
    }
    
    div[data-testid="stExpander"] details {
        border: none !important;
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)) !important;
        color: white !important;
        padding: 1rem 1.5rem !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3)) !important;
    }
    
    div[data-testid="stExpander"] details summary svg {
        color: white !important;
    }
    
    div[data-testid="stExpander"] details > div {
        background: rgba(0, 0, 0, 0.2) !important;
        border: none !important;
        padding: 1rem !important;
    }
    
    /* ===== DATAFRAME STYLING ===== */
    .stDataFrame {
        background: transparent !important;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* ===== DIVIDER STYLING ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* ===== METRIC DISPLAY ===== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-3px);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.5);
    }
    
    /* ===== LOADING ANIMATION ===== */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(102, 126, 234, 0.2);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        animation: loadingPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes loadingPulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* ===== TOOLTIP STYLING ===== */
    .tooltip-container {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip-icon {
        width: 18px;
        height: 18px;
        background: rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        color: white;
        margin-left: 6px;
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        .glass-card {
            padding: 1.5rem;
            border-radius: 16px;
        }
        
        .prediction-card {
            padding: 1.5rem;
        }
        
        .prediction-icon {
            font-size: 3rem;
        }
        
        .prediction-title {
            font-size: 1.8rem;
        }
        
        .stButton > button {
            font-size: 1rem;
            padding: 1rem 2rem;
        }
        
        .result-container {
            flex-direction: column;
            gap: 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.5rem;
        }
        
        .hero-container {
            padding: 1.5rem 1rem;
        }
        
        .card-title {
            font-size: 1.1rem;
        }
    }
    
    /* ===== ANIMATIONS FOR ELEMENTS ===== */
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in-left {
        animation: slideInLeft 0.6s ease-out;
    }
    
    @keyframes slideInLeft {
        0% { opacity: 0; transform: translateX(-30px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    
    .slide-in-right {
        animation: slideInRight 0.6s ease-out;
    }
    
    @keyframes slideInRight {
        0% { opacity: 0; transform: translateX(30px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    
    .section-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }
    
    /* ===== INFO ALERT BOX ===== */
    .info-alert {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin: 1rem 0;
    }
    
    .info-alert-icon {
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    
    .info-alert-text {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.8);
        line-height: 1.5;
    }
    
    .info-alert-text strong {
        color: #a8b5ff;
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
# CALLBACK FUNCTIONS
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
    # ========================================================================
    # HERO HEADER
    # ========================================================================
    st.markdown("""
    <div class="hero-container">
        <div class="floating-badge">
            <span class="badge-dot"></span>
            AI-Powered Prediction System
        </div>
        <h1 class="main-header">👥 Employee Turnover Prediction</h1>
        <p class="sub-header">
            Leverage machine learning to predict employee retention with 
            <span class="sub-header-highlight">precision and confidence</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        return
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">🎛️ Model Dashboard</h2>', unsafe_allow_html=True)
        
        # Model Info Cards
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">🤗 Repository</div>
            <div class="sidebar-card-value">Zlib2/RFC</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">🤖 Algorithm</div>
            <div class="sidebar-card-value">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">⚖️ Class Weight</div>
            <div class="sidebar-card-value">Balanced</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Features List
        st.markdown('<h3 style="color: white; font-size: 1.1rem; margin-bottom: 1rem;">📊 Model Features</h3>', unsafe_allow_html=True)
        
        feature_icons = ["😊", "📅", "⏰", "📁", "📊"]
        feature_names = ["Satisfaction Level", "Years at Company", "Monthly Hours", "Projects Count", "Last Evaluation"]
        
        st.markdown('<ul class="feature-list">', unsafe_allow_html=True)
        for i, (icon, name) in enumerate(zip(feature_icons, feature_names), 1):
            st.markdown(f"""
            <li class="feature-item">
                <span class="feature-number">{i}</span>
                {icon} {name}
            </li>
            """, unsafe_allow_html=True)
        st.markdown('</ul>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Target Variable Info
        st.markdown('<h3 style="color: white; font-size: 1.1rem; margin-bottom: 1rem;">🎯 Prediction Classes</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card" style="border-left: 3px solid #38ef7d;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5rem;">✅</span>
                <div>
                    <div style="color: #38ef7d; font-weight: 600;">Class 0 → Stay</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Likely to remain</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card" style="border-left: 3px solid #f45c43;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <div style="color: #f45c43; font-weight: 600;">Class 1 → Leave</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Likely to depart</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"""
        <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank" 
           style="display: inline-flex; align-items: center; gap: 8px; 
                  background: linear-gradient(135deg, #667eea, #764ba2);
                  color: white; text-decoration: none; padding: 10px 20px;
                  border-radius: 10px; font-weight: 500; font-size: 0.9rem;
                  transition: all 0.3s ease;">
            🔗 View on Hugging Face
        </a>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN INPUT SECTION
    # ========================================================================
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📝</div>
        <h2 class="section-title">Employee Information</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-alert">
        <span class="info-alert-icon">💡</span>
        <span class="info-alert-text">
            <strong>Instructions:</strong> Enter the employee's data below using the sliders and input fields. 
            All fields are required for an accurate prediction.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # INPUT CARDS
    # ========================================================================
    
    # Card 1: Satisfaction & Evaluation
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card-header">
        <div class="card-icon">📊</div>
        <div>
            <h3 class="card-title">Performance Metrics</h3>
            <p class="card-subtitle">Employee satisfaction and evaluation scores</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="input-label">
            <span class="input-label-icon">😊</span>
            Satisfaction Level
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
        
        st.markdown('<p class="input-hint">0 = Very Dissatisfied → 1 = Very Satisfied</p>', unsafe_allow_html=True)
        satisfaction_level = st.session_state.satisfaction_level
    
    with col2:
        st.markdown("""
        <div class="input-label">
            <span class="input-label-icon">📈</span>
            Last Evaluation Score
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
        
        st.markdown('<p class="input-hint">0 = Needs Improvement → 1 = Excellent</p>', unsafe_allow_html=True)
        last_evaluation = st.session_state.last_evaluation
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Card 2: Work Metrics
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card-header">
        <div class="card-icon">⏰</div>
        <div>
            <h3 class="card-title">Work Metrics</h3>
            <p class="card-subtitle">Time, projects, and working hours</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="input-label">
            <span class="input-label-icon">📅</span>
            Years at Company
        </div>
        """, unsafe_allow_html=True)
        time_spend_company = st.number_input(
            "Years at Company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            help="Number of years the employee has worked at the company",
            label_visibility="collapsed"
        )
        st.markdown('<p class="input-hint">Duration of employment</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="input-label">
            <span class="input-label-icon">📁</span>
            Number of Projects
        </div>
        """, unsafe_allow_html=True)
        number_project = st.number_input(
            "Number of Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of projects the employee is currently working on",
            label_visibility="collapsed"
        )
        st.markdown('<p class="input-hint">Active project assignments</p>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="input-label">
            <span class="input-label-icon">⏰</span>
            Avg. Monthly Hours
        </div>
        """, unsafe_allow_html=True)
        average_monthly_hours = st.number_input(
            "Average Monthly Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            help="Average number of hours worked per month",
            label_visibility="collapsed"
        )
        st.markdown('<p class="input-hint">Standard: 160-180 hours</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Generate Prediction", use_container_width=True)
    
    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================
    if predict_button:
        # Show loading animation
        with st.spinner(""):
            st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <p class="loading-text">Analyzing employee data...</p>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5)  # Simulated processing time
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        st.markdown("---")
        
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🎯</div>
            <h2 class="section-title">Prediction Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Results Grid
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-card stay-card">
                    <div class="prediction-icon">✅</div>
                    <h2 class="prediction-title">STAY</h2>
                    <p class="prediction-message">
                        This employee shows strong indicators of <strong>retention</strong>. 
                        They are likely to continue their journey with the company.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-card leave-card">
                    <div class="prediction-icon">⚠️</div>
                    <h2 class="prediction-title">LEAVE</h2>
                    <p class="prediction-message">
                        This employee shows signs of potential <strong>turnover risk</strong>. 
                        Consider intervention strategies to improve retention.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="probability-card">
                <div class="probability-header">
                    📊 Confidence Analysis
                </div>
                
                <div class="prob-item">
                    <div class="prob-label">
                        <span class="prob-text">✅ Probability of Staying</span>
                        <span class="prob-value prob-value-stay">{prob_stay:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar progress-stay" style="width: {prob_stay}%;"></div>
                    </div>
                </div>
                
                <div class="prob-item">
                    <div class="prob-label">
                        <span class="prob-text">⚠️ Probability of Leaving</span>
                        <span class="prob-value prob-value-leave">{prob_leave:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar progress-leave" style="width: {prob_leave}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # INPUT SUMMARY
        # ====================================================================
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.expander("📋 View Input Summary", expanded=False):
                # Display metrics in a nice grid
                st.markdown("""
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-icon">😊</div>
                        <div class="metric-value">{:.2f}</div>
                        <div class="metric-label">Satisfaction</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-icon">📈</div>
                        <div class="metric-value">{:.2f}</div>
                        <div class="metric-label">Evaluation</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-icon">📅</div>
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Years</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-icon">📁</div>
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Projects</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-icon">⏰</div>
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Hours/Month</div>
                    </div>
                </div>
                """.format(
                    satisfaction_level,
                    last_evaluation,
                    time_spend_company,
                    number_project,
                    average_monthly_hours
                ), unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
