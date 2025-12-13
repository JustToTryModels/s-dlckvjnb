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
    /* ===================================================================
       CSS VARIABLES - PROFESSIONAL COLOR PALETTE
       =================================================================== */
    :root {
        --primary-dark: #0f172a;
        --primary-medium: #1e293b;
        --primary-light: #334155;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-teal: #14b8a6;
        --accent-green: #22c55e;
        --accent-red: #ef4444;
        --accent-orange: #f97316;
        --accent-purple: #8b5cf6;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --bg-card: rgba(30, 41, 59, 0.8);
        --bg-glass: rgba(255, 255, 255, 0.05);
        --border-subtle: rgba(255, 255, 255, 0.1);
        --shadow-glow: 0 0 40px rgba(59, 130, 246, 0.15);
        --gradient-primary: linear-gradient(135deg, #3b82f6 0%, #06b6d4 50%, #14b8a6 100%);
        --gradient-success: linear-gradient(135deg, #22c55e 0%, #14b8a6 100%);
        --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
    }

    /* ===================================================================
       GLOBAL STYLES & ANIMATIONS
       =================================================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Keyframe Animations */
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
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.3),
                        0 0 40px rgba(6, 182, 212, 0.2);
        }
        50% { 
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.5),
                        0 0 60px rgba(6, 182, 212, 0.3);
        }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes borderGlow {
        0%, 100% { border-color: rgba(59, 130, 246, 0.5); }
        50% { border-color: rgba(6, 182, 212, 0.8); }
    }
    
    @keyframes textGlow {
        0%, 100% { text-shadow: 0 0 10px rgba(59, 130, 246, 0.5); }
        50% { text-shadow: 0 0 20px rgba(59, 130, 246, 0.8), 0 0 30px rgba(6, 182, 212, 0.5); }
    }

    /* ===================================================================
       MAIN CONTAINER STYLING
       =================================================================== */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-attachment: fixed;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(6, 182, 212, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(139, 92, 246, 0.05) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        position: relative;
        z-index: 1;
    }

    /* ===================================================================
       HEADER STYLES
       =================================================================== */
    .main-header-container {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
        border-radius: 24px;
        border: 1px solid var(--border-subtle);
        backdrop-filter: blur(20px);
        animation: fadeInDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .main-header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 200%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.05),
            transparent
        );
        animation: shimmer 3s infinite;
    }
    
    .logo-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.5));
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        animation: textGlow 3s ease-in-out infinite;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: var(--text-secondary);
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    .header-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        margin-top: 1.5rem;
        background: var(--gradient-primary);
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: pulse-glow 2s ease-in-out infinite;
    }

    /* ===================================================================
       SIDEBAR STYLES
       =================================================================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid var(--border-subtle);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem;
    }
    
    .sidebar-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--accent-blue);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .sidebar-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        animation: fadeInLeft 0.6s ease-out;
    }
    
    .sidebar-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateX(5px);
    }
    
    .sidebar-card-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.75rem;
    }
    
    .sidebar-card-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--text-primary);
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
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--border-subtle);
        transition: all 0.3s ease;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    
    .feature-list-item:last-child {
        border-bottom: none;
    }
    
    .feature-list-item:hover {
        color: var(--text-primary);
        padding-left: 0.5rem;
    }
    
    .feature-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        background: var(--gradient-primary);
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }
    
    .target-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(20, 184, 166, 0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    
    .target-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .target-badge-stay {
        background: var(--accent-green);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .target-badge-leave {
        background: var(--accent-red);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .sidebar-link {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--accent-blue);
        text-decoration: none;
        font-weight: 500;
        padding: 0.75rem 1.25rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 10px;
        transition: all 0.3s ease;
        margin-top: 1rem;
        width: 100%;
        justify-content: center;
    }
    
    .sidebar-link:hover {
        background: rgba(59, 130, 246, 0.2);
        transform: translateY(-2px);
    }

    /* ===================================================================
       SECTION STYLES
       =================================================================== */
    .section-container {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .section-container:hover {
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: var(--shadow-glow);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .section-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        background: var(--gradient-primary);
        border-radius: 12px;
        font-size: 1.5rem;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .section-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    /* ===================================================================
       INPUT FIELD STYLES
       =================================================================== */
    .input-group {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .input-group::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .input-group:hover {
        border-color: rgba(59, 130, 246, 0.4);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .input-group:hover::before {
        opacity: 1;
    }
    
    .input-group:focus-within {
        border-color: var(--accent-blue);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
    }
    
    .input-group:focus-within::before {
        opacity: 1;
    }
    
    .input-label {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    .input-label-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background: var(--bg-glass);
        border-radius: 10px;
        font-size: 1.2rem;
    }
    
    .input-tooltip {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.75rem;
        padding: 0.75rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        border-left: 3px solid var(--accent-blue);
    }
    
    /* Streamlit Input Overrides */
    .stSlider > div > div > div {
        background: var(--border-subtle) !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--gradient-primary) !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid var(--accent-blue) !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        transition: all 0.2s ease !important;
    }
    
    .stSlider > div > div > div > div > div:hover {
        transform: scale(1.2);
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.7);
    }
    
    .stNumberInput > div > div > input {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stNumberInput > div > div > input:hover {
        border-color: rgba(59, 130, 246, 0.5) !important;
    }

    /* ===================================================================
       PREDICT BUTTON STYLES
       =================================================================== */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 50%, #14b8a6 100%);
        background-size: 200% 200%;
        color: white !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 1.25rem 2.5rem !important;
        border-radius: 16px !important;
        border: none !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 10px 30px rgba(59, 130, 246, 0.4),
            0 0 50px rgba(6, 182, 212, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        animation: gradientFlow 4s ease infinite, pulse-glow 2s ease-in-out infinite;
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
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #06b6d4 0%, #14b8a6 50%, #3b82f6 100%);
        background-size: 200% 200%;
        transform: translateY(-4px) scale(1.02);
        box-shadow: 
            0 15px 40px rgba(59, 130, 246, 0.5),
            0 0 80px rgba(6, 182, 212, 0.3),
            inset 0 0 20px rgba(255, 255, 255, 0.1);
        animation: gradientFlow 2s ease infinite;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(2px) scale(0.98) !important;
        box-shadow: 
            0 5px 20px rgba(59, 130, 246, 0.4),
            inset 0 0 30px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:focus {
        outline: none !important;
        box-shadow: 
            0 0 0 3px rgba(59, 130, 246, 0.3),
            0 10px 30px rgba(59, 130, 246, 0.4);
    }
    
    .stButton > button:focus:not(:focus-visible) {
        outline: none !important;
    }

    /* ===================================================================
       PREDICTION RESULT STYLES
       =================================================================== */
    .result-container {
        animation: scaleIn 0.5s ease-out;
    }
    
    .prediction-card {
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        opacity: 0.1;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.4'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }
    
    .prediction-stay {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(20, 184, 166, 0.2) 100%);
        border: 2px solid rgba(34, 197, 94, 0.5);
        box-shadow: 
            0 20px 40px rgba(34, 197, 94, 0.2),
            inset 0 0 60px rgba(34, 197, 94, 0.1);
    }
    
    .prediction-leave {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(249, 115, 22, 0.2) 100%);
        border: 2px solid rgba(239, 68, 68, 0.5);
        box-shadow: 
            0 20px 40px rgba(239, 68, 68, 0.2),
            inset 0 0 60px rgba(239, 68, 68, 0.1);
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: float 2s ease-in-out infinite;
    }
    
    .prediction-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 4px;
    }
    
    .prediction-stay .prediction-title {
        background: linear-gradient(135deg, #22c55e 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .prediction-leave .prediction-title {
        background: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .prediction-message {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    /* Probability Section */
    .probability-section {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2rem;
        animation: fadeInUp 0.7s ease-out;
    }
    
    .probability-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
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
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    .probability-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    .probability-value-stay {
        color: var(--accent-green);
    }
    
    .probability-value-leave {
        color: var(--accent-red);
    }
    
    .progress-container {
        height: 16px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255, 255, 255, 0.3) 50%,
            transparent 100%
        );
        animation: shimmer 2s infinite;
    }
    
    .progress-stay {
        background: linear-gradient(90deg, #22c55e 0%, #14b8a6 100%);
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.5);
    }
    
    .progress-leave {
        background: linear-gradient(90deg, #ef4444 0%, #f97316 100%);
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    }

    /* ===================================================================
       EXPANDER / SUMMARY STYLES
       =================================================================== */
    div[data-testid="stExpander"] {
        border: none !important;
        background: transparent;
    }
    
    div[data-testid="stExpander"] details {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stExpander"] details:hover {
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%);
        color: var(--text-primary) !important;
        border-radius: 16px;
        padding: 1rem 1.5rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(6, 182, 212, 0.3) 100%);
    }
    
    div[data-testid="stExpander"] details summary svg {
        color: var(--accent-cyan) !important;
    }
    
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 16px 16px 0 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    div[data-testid="stExpander"] details > div {
        background: var(--bg-glass);
        border-top: none !important;
        padding: 1.5rem;
    }

    /* ===================================================================
       DATA TABLE STYLES
       =================================================================== */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame > div > div > div > div {
        background: var(--bg-glass) !important;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    .stDataFrame td {
        background: var(--bg-glass) !important;
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.875rem 1rem !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }
    
    .stDataFrame tr:hover td {
        background: rgba(59, 130, 246, 0.1) !important;
        color: var(--text-primary) !important;
    }

    /* ===================================================================
       LOADING ANIMATION STYLES
       =================================================================== */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        animation: fadeIn 0.3s ease;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid var(--border-subtle);
        border-top-color: var(--accent-blue);
        border-radius: 50%;
        animation: rotate 1s linear infinite;
    }
    
    .loading-text {
        margin-top: 1.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: var(--text-secondary);
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ===================================================================
       TOOLTIP STYLES
       =================================================================== */
    .tooltip-container {
        position: relative;
        display: inline-block;
    }
    
    .tooltip-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        background: var(--accent-blue);
        border-radius: 50%;
        font-size: 0.75rem;
        color: white;
        cursor: help;
        margin-left: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .tooltip-icon:hover {
        background: var(--accent-cyan);
        transform: scale(1.1);
    }

    /* ===================================================================
       DIVIDER STYLES
       =================================================================== */
    .custom-divider {
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent 0%,
            var(--border-subtle) 20%,
            var(--accent-blue) 50%,
            var(--border-subtle) 80%,
            transparent 100%
        );
        margin: 2rem 0;
        animation: fadeIn 0.5s ease;
    }

    /* ===================================================================
       RESPONSIVE STYLES
       =================================================================== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        .main .block-container {
            padding: 1rem;
        }
        
        .section-container {
            padding: 1.25rem;
        }
        
        .input-group {
            padding: 1rem;
        }
        
        .prediction-card {
            padding: 1.5rem;
        }
        
        .prediction-title {
            font-size: 1.75rem;
        }
        
        .prediction-icon {
            font-size: 3rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.5rem;
            letter-spacing: 0;
        }
        
        .logo-icon {
            font-size: 2.5rem;
        }
        
        .header-badge {
            font-size: 0.7rem;
            padding: 0.4rem 1rem;
        }
        
        .stButton > button {
            font-size: 1rem !important;
            padding: 1rem 1.5rem !important;
        }
    }

    /* ===================================================================
       METRIC DISPLAY STYLES
       =================================================================== */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-item {
        flex: 1;
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    /* ===================================================================
       INFO BOX STYLES
       =================================================================== */
    .info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }
    
    .info-box-icon {
        font-size: 1.5rem;
        flex-shrink: 0;
    }
    
    .info-box-content {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    .info-box-content strong {
        color: var(--text-primary);
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        "name": "Satisfaction Level",
        "description": "Employee's self-reported job satisfaction score",
        "range": "0 (Very Dissatisfied) → 1 (Very Satisfied)"
    },
    "time_spend_company": {
        "icon": "📅",
        "name": "Years at Company",
        "description": "Total number of years the employee has worked here",
        "range": "1 - 40 years"
    },
    "average_monthly_hours": {
        "icon": "⏰",
        "name": "Average Monthly Hours",
        "description": "Average hours worked per month",
        "range": "80 - 350 hours"
    },
    "number_project": {
        "icon": "📁",
        "name": "Number of Projects",
        "description": "Number of projects assigned to the employee",
        "range": "1 - 10 projects"
    },
    "last_evaluation": {
        "icon": "📊",
        "name": "Last Evaluation Score",
        "description": "Most recent performance evaluation score",
        "range": "0 (Poor) → 1 (Excellent)"
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
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    st.markdown("""
    <div class="main-header-container">
        <div class="logo-icon">👥</div>
        <h1 class="main-header">Employee Turnover Prediction</h1>
        <p class="sub-header">Leverage AI to predict employee retention and make data-driven HR decisions</p>
        <div class="header-badge">🤖 Powered by Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)
    
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
        st.markdown('<div class="sidebar-header">ℹ️ Model Information</div>', unsafe_allow_html=True)
        
        # Model Details Card
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">🤗 Repository</div>
            <div class="sidebar-card-value">{HF_REPO_ID}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">🤖 Algorithm</div>
            <div class="sidebar-card-value">Random Forest Classifier</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">⚖️ Class Weight</div>
            <div class="sidebar-card-value">Balanced</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features List
        st.markdown('<div class="sidebar-header" style="margin-top: 1rem;">📋 Model Features</div>', unsafe_allow_html=True)
        
        features_html = '<ul class="feature-list">'
        for i, feat in enumerate(BEST_FEATURES, 1):
            info = FEATURE_INFO[feat]
            features_html += f'''
            <li class="feature-list-item">
                <span class="feature-number">{i}</span>
                <span>{info["icon"]} {info["name"]}</span>
            </li>
            '''
        features_html += '</ul>'
        st.markdown(features_html, unsafe_allow_html=True)
        
        # Target Variable
        st.markdown('<div class="sidebar-header" style="margin-top: 1rem;">🎯 Prediction Classes</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="target-box">
            <div class="target-item">
                <span class="target-badge-stay">0</span>
                <span style="color: var(--text-secondary);">→ Employee likely to <strong style="color: var(--accent-green);">STAY</strong></span>
            </div>
            <div class="target-item">
                <span class="target-badge-leave">1</span>
                <span style="color: var(--text-secondary);">→ Employee likely to <strong style="color: var(--accent-red);">LEAVE</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Link to HF
        st.markdown(f"""
        <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank" class="sidebar-link">
            🔗 View on Hugging Face
        </a>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN INPUT SECTION
    # ========================================================================
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-container">
        <div class="section-header">
            <div class="section-icon">📝</div>
            <div>
                <h2 class="section-title">Employee Information</h2>
                <p class="section-subtitle">Enter the employee's details below to generate a turnover prediction</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <div class="info-box-icon">💡</div>
        <div class="info-box-content">
            <strong>How it works:</strong> Fill in the employee metrics below. The AI model will analyze 
            these factors to predict whether the employee is at risk of leaving the company.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # INPUT FIELDS - ROW 1
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">😊</span>
                <span>Satisfaction Level</span>
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
        
        st.markdown("""
            <div class="input-tooltip">
                📌 0 = Very Dissatisfied → 1 = Very Satisfied
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        satisfaction_level = st.session_state.satisfaction_level
    
    with col2:
        st.markdown(f"""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">📊</span>
                <span>Last Evaluation Score</span>
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
        
        st.markdown("""
            <div class="input-tooltip">
                📌 0 = Poor Performance → 1 = Excellent Performance
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        last_evaluation = st.session_state.last_evaluation
    
    # ========================================================================
    # INPUT FIELDS - ROW 2
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">📅</span>
                <span>Years at Company</span>
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
        
        st.markdown("""
            <div class="input-tooltip">
                📌 Total years the employee has been with the company
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">📁</span>
                <span>Number of Projects</span>
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
        
        st.markdown("""
            <div class="input-tooltip">
                📌 Current number of projects assigned to the employee
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # INPUT FIELDS - ROW 3
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">⏰</span>
                <span>Average Monthly Hours</span>
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
        
        st.markdown("""
            <div class="input-tooltip">
                📌 Average hours worked per month (typical range: 150-250)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮  Generate Prediction", use_container_width=True)
    
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
            time.sleep(1.5)  # Simulate processing
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Results Section Header
        st.markdown("""
        <div class="section-container result-container">
            <div class="section-header">
                <div class="section-icon">🎯</div>
                <div>
                    <h2 class="section-title">Prediction Results</h2>
                    <p class="section-subtitle">AI-powered analysis of employee turnover risk</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Results columns
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.markdown(f"""
                <div class="prediction-card prediction-stay">
                    <div class="prediction-icon">✅</div>
                    <h2 class="prediction-title">STAY</h2>
                    <p class="prediction-message">
                        This employee is likely to <strong>remain</strong> with the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card prediction-leave">
                    <div class="prediction-icon">⚠️</div>
                    <h2 class="prediction-title">LEAVE</h2>
                    <p class="prediction-message">
                        This employee is at <strong>risk of leaving</strong> the company
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="probability-section">
                <div class="probability-header">
                    📊 Confidence Probabilities
                </div>
                
                <div class="probability-item">
                    <div class="probability-label">
                        <span class="probability-text">Probability of Staying</span>
                        <span class="probability-value probability-value-stay">{prob_stay:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar progress-stay" style="width: {prob_stay}%;"></div>
                    </div>
                </div>
                
                <div class="probability-item">
                    <div class="probability-label">
                        <span class="probability-text">Probability of Leaving</span>
                        <span class="probability-value probability-value-leave">{prob_leave:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar progress-leave" style="width: {prob_leave}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # INPUT SUMMARY (Collapsible)
        # ====================================================================
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
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
                        f"{average_monthly_hours} hours"
                    ]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Recommendations based on prediction
                if prediction == 1:
                    st.markdown("""
                    <div class="info-box" style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(249, 115, 22, 0.1) 100%); border-color: 
