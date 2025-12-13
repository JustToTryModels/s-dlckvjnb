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
# COMPREHENSIVE CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* ===== GOOGLE FONTS IMPORT ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== ROOT VARIABLES ===== */
    :root {
        --primary-color: #1E3A5F;
        --primary-light: #2E5A8F;
        --primary-dark: #0D1F33;
        --accent-color: #00D4AA;
        --accent-hover: #00F5C4;
        --success-color: #10B981;
        --success-light: #D1FAE5;
        --danger-color: #EF4444;
        --danger-light: #FEE2E2;
        --warning-color: #F59E0B;
        --warning-light: #FEF3C7;
        --text-primary: #1F2937;
        --text-secondary: #6B7280;
        --text-muted: #9CA3AF;
        --bg-primary: #F8FAFC;
        --bg-secondary: #FFFFFF;
        --bg-tertiary: #F1F5F9;
        --border-color: #E2E8F0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --transition-fast: 0.15s ease;
        --transition-normal: 0.3s ease;
        --transition-slow: 0.5s ease;
        --border-radius-sm: 6px;
        --border-radius-md: 10px;
        --border-radius-lg: 16px;
        --border-radius-xl: 24px;
    }
    
    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 50%, #F0FDF4 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== ANIMATIONS ===== */
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
    
    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes fadeInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes pulse {
        0%, 100% { 
            box-shadow: 0 0 0 0 rgba(0, 212, 170, 0.4);
        }
        50% { 
            box-shadow: 0 0 0 15px rgba(0, 212, 170, 0);
        }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes bounce {
        0%, 20%, 53%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-20px); }
        60% { transform: translateY(-10px); }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes ripple {
        0% {
            transform: scale(0);
            opacity: 1;
        }
        100% {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    @keyframes slideInFromBottom {
        0% {
            transform: translateY(100%);
            opacity: 0;
        }
        100% {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 0 20px rgba(0, 212, 170, 0.3),
                        0 0 40px rgba(0, 212, 170, 0.2),
                        0 0 60px rgba(0, 212, 170, 0.1);
        }
        50% {
            box-shadow: 0 0 30px rgba(0, 212, 170, 0.5),
                        0 0 60px rgba(0, 212, 170, 0.3),
                        0 0 90px rgba(0, 212, 170, 0.2);
        }
    }
    
    /* ===== HEADER STYLES ===== */
    .hero-container {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 50%, #3B82F6 100%);
        border-radius: var(--border-radius-xl);
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        animation: fadeInDown 0.8s ease-out;
        box-shadow: var(--shadow-xl);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
    }
    
    .hero-container::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    .hero-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: bounce 2s ease infinite;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
    }
    
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 4px 8px rgba(0, 0, 0, 0.2);
        position: relative;
        z-index: 1;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1.25rem;
        border-radius: 50px;
        color: white;
        font-size: 0.875rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        transition: all var(--transition-normal);
    }
    
    .hero-badge:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
    }
    
    /* ===== SIDEBAR STYLES ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.15);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
        font-family: 'Poppins', sans-serif;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
    }
    
    .sidebar-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--border-radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all var(--transition-normal);
        animation: fadeInLeft 0.6s ease-out;
    }
    
    .sidebar-card:hover {
        background: rgba(255, 255, 255, 0.12);
        transform: translateX(5px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .sidebar-card-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--accent-color);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-card-value {
        font-size: 1.1rem;
        font-weight: 500;
        color: white;
        line-height: 1.5;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .feature-list-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: var(--border-radius-md);
        margin-bottom: 0.5rem;
        color: white;
        font-size: 0.95rem;
        transition: all var(--transition-normal);
        border: 1px solid transparent;
    }
    
    .feature-list-item:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
    
    .feature-number {
        width: 24px;
        height: 24px;
        background: var(--accent-color);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--primary-dark);
        flex-shrink: 0;
    }
    
    .sidebar-link {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        background: var(--accent-color);
        color: var(--primary-dark) !important;
        text-decoration: none;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all var(--transition-normal);
        margin-top: 1rem;
    }
    
    .sidebar-link:hover {
        background: var(--accent-hover);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.4);
    }
    
    /* ===== SECTION STYLES ===== */
    .section-container {
        background: var(--bg-secondary);
        border-radius: var(--border-radius-xl);
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border-color);
        animation: fadeInUp 0.6s ease-out;
        transition: all var(--transition-normal);
    }
    
    .section-container:hover {
        box-shadow: var(--shadow-xl);
        transform: translateY(-2px);
    }
    
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-title-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
        border-radius: var(--border-radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: var(--shadow-md);
    }
    
    .section-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    /* ===== INPUT CARD STYLES ===== */
    .input-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-lg);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all var(--transition-normal);
        position: relative;
        overflow: hidden;
    }
    
    .input-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--primary-color), var(--accent-color));
        border-radius: 4px 0 0 4px;
    }
    
    .input-card:hover {
        border-color: var(--primary-light);
        box-shadow: var(--shadow-lg);
        transform: translateY(-3px);
    }
    
    .input-card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .input-card-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
        border-radius: var(--border-radius-sm);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        box-shadow: var(--shadow-sm);
    }
    
    .input-card-title {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1rem;
    }
    
    .input-card-subtitle {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.15rem;
    }
    
    .input-tooltip {
        background: var(--bg-tertiary);
        border-radius: var(--border-radius-sm);
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.75rem;
        border-left: 3px solid var(--accent-color);
        line-height: 1.5;
    }
    
    /* ===== SLIDER STYLES ===== */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color)) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid var(--primary-color) !important;
        box-shadow: var(--shadow-md) !important;
        transition: all var(--transition-fast) !important;
    }
    
    .stSlider > div > div > div > div > div:hover {
        transform: scale(1.2) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    /* ===== NUMBER INPUT STYLES ===== */
    .stNumberInput > div > div > input {
        border-radius: var(--border-radius-md) !important;
        border: 2px solid var(--border-color) !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        transition: all var(--transition-normal) !important;
        background: white !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.15) !important;
    }
    
    .stNumberInput > div > div > input:hover {
        border-color: var(--primary-light) !important;
    }
    
    /* ===== PREDICT BUTTON STYLES ===== */
    .predict-button-container {
        padding: 2rem 0;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 50%, #3B82F6 100%);
        background-size: 200% 200%;
        color: white !important;
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem;
        font-weight: 700 !important;
        padding: 1.25rem 3rem;
        border-radius: 60px;
        border: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(30, 58, 95, 0.3),
                    0 5px 15px rgba(30, 58, 95, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: gradientFlow 4s ease infinite;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(30, 58, 95, 0.4),
                    0 8px 20px rgba(30, 58, 95, 0.3),
                    0 0 60px rgba(0, 212, 170, 0.2);
        background-size: 200% 200%;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(0.98);
        box-shadow: 0 5px 20px rgba(30, 58, 95, 0.3);
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
        transition: left 0.7s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button::after {
        content: '🔮';
        position: absolute;
        right: 25px;
        font-size: 1.5rem;
        animation: float 2s ease-in-out infinite;
    }
    
    .stButton > button:focus,
    .stButton > button:focus-visible {
        outline: none !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(30, 58, 95, 0.3),
                    0 5px 15px rgba(30, 58, 95, 0.2),
                    0 0 0 4px rgba(30, 58, 95, 0.2);
    }
    
    /* ===== RESULT BOX STYLES ===== */
    .result-container {
        animation: scaleIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .prediction-box {
        padding: 2.5rem;
        border-radius: var(--border-radius-xl);
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all var(--transition-normal);
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23000000' fill-opacity='0.03' fill-rule='evenodd'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E");
    }
    
    .stay-prediction {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border: 2px solid var(--success-color);
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.2),
                    inset 0 -5px 20px rgba(16, 185, 129, 0.1);
    }
    
    .stay-prediction:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(16, 185, 129, 0.3),
                    inset 0 -5px 20px rgba(16, 185, 129, 0.15);
    }
    
    .leave-prediction {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 2px solid var(--danger-color);
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.2),
                    inset 0 -5px 20px rgba(239, 68, 68, 0.1);
    }
    
    .leave-prediction:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(239, 68, 68, 0.3),
                    inset 0 -5px 20px rgba(239, 68, 68, 0.15);
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: bounce 1s ease;
        position: relative;
        z-index: 1;
    }
    
    .prediction-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .stay-prediction .prediction-title {
        color: #047857;
    }
    
    .leave-prediction .prediction-title {
        color: #B91C1C;
    }
    
    .prediction-subtitle {
        font-size: 1.15rem;
        position: relative;
        z-index: 1;
        line-height: 1.6;
    }
    
    .stay-prediction .prediction-subtitle {
        color: #065F46;
    }
    
    .leave-prediction .prediction-subtitle {
        color: #991B1B;
    }
    
    /* ===== PROBABILITY CARD STYLES ===== */
    .probability-card {
        background: white;
        border-radius: var(--border-radius-lg);
        padding: 2rem;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border-color);
        height: 100%;
    }
    
    .probability-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .probability-item {
        margin-bottom: 1.5rem;
    }
    
    .probability-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .probability-text {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .probability-value {
        font-weight: 700;
        font-size: 1.25rem;
    }
    
    .probability-value.stay {
        color: var(--success-color);
    }
    
    .probability-value.leave {
        color: var(--danger-color);
    }
    
    .progress-bar-container {
        width: 100%;
        background: var(--bg-tertiary);
        border-radius: 50px;
        height: 16px;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 50px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.3),
            transparent
        );
        animation: shimmer 2s infinite;
        background-size: 200% 100%;
    }
    
    .progress-bar-green {
        background: linear-gradient(90deg, #10B981, #34D399);
    }
    
    .progress-bar-red {
        background: linear-gradient(90deg, #EF4444, #F87171);
    }
    
    /* ===== EXPANDER STYLES ===== */
    div[data-testid="stExpander"] {
        border: none !important;
        border-radius: var(--border-radius-lg) !important;
        overflow: hidden;
        box-shadow: var(--shadow-md);
        animation: fadeInUp 0.6s ease-out;
    }
    
    div[data-testid="stExpander"] details {
        border: none !important;
        background: white;
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%) !important;
        color: white !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all var(--transition-normal) !important;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        background: linear-gradient(135deg, var(--primary-light) 0%, #3B82F6 100%) !important;
    }
    
    div[data-testid="stExpander"] details summary svg {
        color: white !important;
        fill: white !important;
    }
    
    div[data-testid="stExpander"] details > div {
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        padding: 1.5rem !important;
    }
    
    /* ===== DATA TABLE STYLES ===== */
    .stDataFrame {
        border-radius: var(--border-radius-md) !important;
        overflow: hidden !important;
    }
    
    .stDataFrame table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
    }
    
    .stDataFrame th {
        background: var(--primary-color) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    
    .stDataFrame td {
        padding: 0.875rem 1rem !important;
        border-bottom: 1px solid var(--border-color) !important;
        transition: background var(--transition-fast);
    }
    
    .stDataFrame tr:hover td {
        background: var(--bg-tertiary) !important;
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
        border: 4px solid var(--bg-tertiary);
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    .loading-text {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 500;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* ===== INFO ALERT STYLES ===== */
    .info-alert {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border: 1px solid #C7D2FE;
        border-radius: var(--border-radius-lg);
        padding: 1.25rem 1.5rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin: 1rem 0;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .info-alert-icon {
        font-size: 1.5rem;
        flex-shrink: 0;
    }
    
    .info-alert-content {
        flex: 1;
    }
    
    .info-alert-title {
        font-weight: 600;
        color: #3730A3;
        margin-bottom: 0.25rem;
    }
    
    .info-alert-text {
        color: #4338CA;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* ===== DIVIDER STYLES ===== */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
        border: none;
    }
    
    /* ===== METRIC STYLES ===== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-item {
        background: white;
        border-radius: var(--border-radius-md);
        padding: 1rem;
        text-align: center;
        border: 1px solid var(--border-color);
        transition: all var(--transition-normal);
    }
    
    .metric-item:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* ===== RESPONSIVE STYLES ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        .hero-container {
            padding: 2rem 1.5rem;
        }
        
        .section-container {
            padding: 1.5rem;
        }
        
        .prediction-title {
            font-size: 2rem;
        }
        
        .stButton > button {
            font-size: 1rem;
            padding: 1rem 2rem;
        }
    }
    
    /* ===== TOOLTIP STYLES ===== */
    .custom-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .custom-tooltip .tooltip-text {
        visibility: hidden;
        width: 250px;
        background: var(--primary-dark);
        color: white;
        text-align: left;
        border-radius: var(--border-radius-md);
        padding: 1rem;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: all var(--transition-normal);
        font-size: 0.85rem;
        line-height: 1.5;
        box-shadow: var(--shadow-xl);
    }
    
    .custom-tooltip .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -8px;
        border-width: 8px;
        border-style: solid;
        border-color: var(--primary-dark) transparent transparent transparent;
    }
    
    .custom-tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
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

FEATURE_INFO = {
    "satisfaction_level": {
        "icon": "😊",
        "title": "Satisfaction Level",
        "subtitle": "Employee happiness index",
        "tooltip": "Measures how satisfied the employee is with their job, workplace environment, and overall experience. Higher values indicate greater satisfaction."
    },
    "time_spend_company": {
        "icon": "📅",
        "title": "Years at Company",
        "subtitle": "Tenure duration",
        "tooltip": "The number of years the employee has been working at the company. Tenure can significantly impact turnover decisions."
    },
    "average_monthly_hours": {
        "icon": "⏰",
        "title": "Average Monthly Hours",
        "subtitle": "Work hours per month",
        "tooltip": "Average number of hours worked per month. Both too few and too many hours can indicate potential turnover risk."
    },
    "number_project": {
        "icon": "📁",
        "title": "Number of Projects",
        "subtitle": "Active project count",
        "tooltip": "The number of projects the employee is currently assigned to. Overload or underutilization can affect turnover."
    },
    "last_evaluation": {
        "icon": "📊",
        "title": "Last Evaluation Score",
        "subtitle": "Performance rating",
        "tooltip": "The employee's most recent performance evaluation score. This reflects their work quality and contribution to the company."
    }
}

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
# CALLBACK FUNCTIONS FOR SYNCING
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
    # HERO SECTION
    # ========================================================================
    st.markdown("""
    <div class="hero-container">
        <span class="hero-icon">👥</span>
        <h1 class="main-header">Employee Turnover Prediction</h1>
        <p class="sub-header">Leverage machine learning to predict whether an employee is at risk of leaving your organization. Make data-driven decisions to improve retention.</p>
        <div class="badge-container">
            <span class="hero-badge">🤖 AI-Powered</span>
            <span class="hero-badge">📊 95% Accuracy</span>
            <span class="hero-badge">⚡ Real-time Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model silently
    model = load_model_from_huggingface()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the Hugging Face repository.")
        st.markdown(f"""
        <div class="info-alert">
            <span class="info-alert-icon">💡</span>
            <div class="info-alert-content">
                <div class="info-alert-title">Repository Information</div>
                <div class="info-alert-text">
                    The model is hosted at: <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank">https://huggingface.co/{HF_REPO_ID}</a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span style="font-size: 2.5rem;">🧠</span>
            <h2 style="margin-top: 0.5rem; margin-bottom: 0;">Model Info</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Information Cards
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                <span>🤗</span> Repository
            </div>
            <div class="sidebar-card-value">{HF_REPO_ID}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                <span>🤖</span> Algorithm
            </div>
            <div class="sidebar-card-value">Random Forest Classifier</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                <span>📊</span> Features Used
            </div>
            <div class="sidebar-card-value">{len(BEST_FEATURES)} Key Predictors</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                <span>⚖️</span> Class Weight
            </div>
            <div class="sidebar-card-value">Balanced</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature List
        st.markdown("""
        <h3 style="color: white; margin-bottom: 1rem;">📋 Selected Features</h3>
        """, unsafe_allow_html=True)
        
        features_html = '<div class="feature-list">'
        for i, feat in enumerate(BEST_FEATURES, 1):
            info = FEATURE_INFO.get(feat, {})
            features_html += f'''
            <div class="feature-list-item">
                <span class="feature-number">{i}</span>
                <span>{info.get("icon", "📌")} {info.get("title", feat.replace("_", " ").title())}</span>
            </div>
            '''
        features_html += '</div>'
        st.markdown(features_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Target Variable Info
        st.markdown("""
        <h3 style="color: white; margin-bottom: 1rem;">🎯 Target Classes</h3>
        <div class="sidebar-card">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">✅</span>
                <div>
                    <div style="font-weight: 600; color: #10B981;">0 → STAY</div>
                    <div style="font-size: 0.85rem; color: rgba(255,255,255,0.7);">Employee likely to remain</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <div style="font-weight: 600; color: #EF4444;">1 → LEAVE</div>
                    <div style="font-size: 0.85rem; color: rgba(255,255,255,0.7);">Employee at risk of leaving</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Link to Hugging Face
        st.markdown(f"""
        <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank" class="sidebar-link">
            🔗 View on Hugging Face
        </a>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN INPUT SECTION
    # ========================================================================
    st.markdown("""
    <div class="section-container">
        <div class="section-title">
            <div class="section-title-icon">📝</div>
            Enter Employee Information
        </div>
        <p class="section-subtitle">
            Provide the employee's key metrics below to generate a turnover risk prediction. 
            All fields are required for accurate results.
        </p>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # INPUT GRID - Row 1
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        info = FEATURE_INFO["satisfaction_level"]
        st.markdown(f"""
        <div class="input-card">
            <div class="input-card-header">
                <div class="input-card-icon">{info["icon"]}</div>
                <div>
                    <div class="input-card-title">{info["title"]}</div>
                    <div class="input-card-subtitle">{info["subtitle"]}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        sat_col1, sat_col2 = st.columns([3, 1])
        with sat_col1:
            st.slider(
                "Satisfaction",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.satisfaction_level,
                step=0.01,
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
        
        st.markdown(f"""
            <div class="input-tooltip">
                💡 {info["tooltip"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        satisfaction_level = st.session_state.satisfaction_level
    
    with col2:
        info = FEATURE_INFO["last_evaluation"]
        st.markdown(f"""
        <div class="input-card">
            <div class="input-card-header">
                <div class="input-card-icon">{info["icon"]}</div>
                <div>
                    <div class="input-card-title">{info["title"]}</div>
                    <div class="input-card-subtitle">{info["subtitle"]}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        eval_col1, eval_col2 = st.columns([3, 1])
        with eval_col1:
            st.slider(
                "Evaluation",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.last_evaluation,
                step=0.01,
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
        
        st.markdown(f"""
            <div class="input-tooltip">
                💡 {info["tooltip"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        last_evaluation = st.session_state.last_evaluation
    
    # ========================================================================
    # INPUT GRID - Row 2
    # ========================================================================
    col3, col4 = st.columns(2)
    
    with col3:
        info = FEATURE_INFO["time_spend_company"]
        st.markdown(f"""
        <div class="input-card">
            <div class="input-card-header">
                <div class="input-card-icon">{info["icon"]}</div>
                <div>
                    <div class="input-card-title">{info["title"]}</div>
                    <div class="input-card-subtitle">{info["subtitle"]}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        time_spend_company = st.number_input(
            "Years at Company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            label_visibility="collapsed"
        )
        
        st.markdown(f"""
            <div class="input-tooltip">
                💡 {info["tooltip"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        info = FEATURE_INFO["number_project"]
        st.markdown(f"""
        <div class="input-card">
            <div class="input-card-header">
                <div class="input-card-icon">{info["icon"]}</div>
                <div>
                    <div class="input-card-title">{info["title"]}</div>
                    <div class="input-card-subtitle">{info["subtitle"]}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        number_project = st.number_input(
            "Number of Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            label_visibility="collapsed"
        )
        
        st.markdown(f"""
            <div class="input-tooltip">
                💡 {info["tooltip"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # INPUT GRID - Row 3
    # ========================================================================
    col5, col6 = st.columns(2)
    
    with col5:
        info = FEATURE_INFO["average_monthly_hours"]
        st.markdown(f"""
        <div class="input-card">
            <div class="input-card-header">
                <div class="input-card-icon">{info["icon"]}</div>
                <div>
                    <div class="input-card-title">{info["title"]}</div>
                    <div class="input-card-subtitle">{info["subtitle"]}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        average_monthly_hours = st.number_input(
            "Average Monthly Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            label_visibility="collapsed"
        )
        
        st.markdown(f"""
            <div class="input-tooltip">
                💡 {info["tooltip"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close section-container
    
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
    st.markdown('<div class="predict-button-container">', unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Analyze Turnover Risk", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================
    if predict_button:
        # Show loading animation
        with st.spinner(""):
            st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">Analyzing employee data...</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)  # Brief delay for visual effect
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        # Results Section
        st.markdown("""
        <div class="section-container result-container">
            <div class="section-title">
                <div class="section-title-icon">🎯</div>
                Prediction Results
            </div>
        """, unsafe_allow_html=True)
        
        result_col1, result_col2 = st.columns([1.2, 1])
        
        with result_col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-box stay-prediction">
                    <span class="prediction-icon">✅</span>
                    <h2 class="prediction-title">LIKELY TO STAY</h2>
                    <p class="prediction-subtitle">
                        Based on the provided metrics, this employee shows strong indicators of retention. 
                        They appear engaged and satisfied with their position.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box leave-prediction">
                    <span class="prediction-icon">⚠️</span>
                    <h2 class="prediction-title">AT RISK OF LEAVING</h2>
                    <p class="prediction-subtitle">
                        Warning: This employee shows signs of potential turnover. 
                        Consider proactive engagement and retention strategies.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown(f"""
            <div class="probability-card">
                <div class="probability-title">📊 Confidence Analysis</div>
                
                <div class="probability-item">
                    <div class="probability-label">
                        <span class="probability-text">✅ Probability of Staying</span>
                        <span class="probability-value stay">{prob_stay:.1f}%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill progress-bar-green" style="width: {prob_stay}%;"></div>
                    </div>
                </div>
                
                <div class="probability-item">
                    <div class="probability-label">
                        <span class="probability-text">⚠️ Probability of Leaving</span>
                        <span class="probability-value leave">{prob_leave:.1f}%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill progress-bar-red" style="width: {prob_leave}%;"></div>
                    </div>
                </div>
                
                <div class="input-tooltip" style="margin-top: 1.5rem;">
                    💡 Higher confidence indicates stronger prediction reliability. 
                    Consider additional factors for borderline cases.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close result section
        
        # ====================================================================
        # INPUT SUMMARY
        # ====================================================================
        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
        
        col_exp1, col_exp2, col_exp3 = st.columns([1, 2, 1])
        with col_exp2:
            with st.expander("📋 View Input Summary", expanded=False):
                summary_data = []
                for feat in BEST_FEATURES:
                    info = FEATURE_INFO.get(feat, {})
                    value = input_data[feat]
                    
                    if feat in ["satisfaction_level", "last_evaluation"]:
                        formatted_value = f"{value:.2f}"
                    elif feat == "time_spend_company":
                        formatted_value = f"{value} years"
                    elif feat == "number_project":
                        formatted_value = f"{value} projects"
                    else:
                        formatted_value = f"{value} hours"
                    
                    summary_data.append({
                        "Feature": f"{info.get('icon', '📌')} {info.get('title', feat)}",
                        "Value": formatted_value
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
