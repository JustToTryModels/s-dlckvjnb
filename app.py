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
# ENHANCED PROFESSIONAL CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* ===================================================================
       ROOT VARIABLES & THEME
    =================================================================== */
    :root {
        --primary-navy: #0F1C3F;
        --primary-navy-light: #1A2F5A;
        --accent-blue: #2E7D9E;
        --accent-cyan: #00D4FF;
        --accent-purple: #7C3AED;
        --accent-pink: #EC4899;
        --success-green: #10B981;
        --success-light: #34D399;
        --warning-orange: #F59E0B;
        --danger-red: #EF4444;
        --danger-light: #F87171;
        --bg-dark: #0A0F1C;
        --bg-card: rgba(15, 28, 63, 0.6);
        --bg-glass: rgba(255, 255, 255, 0.05);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --border-subtle: rgba(255, 255, 255, 0.1);
        --shadow-glow: 0 0 40px rgba(46, 125, 158, 0.3);
    }
    
    /* ===================================================================
       GLOBAL STYLES & ANIMATIONS
    =================================================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0A0F1C 0%, #1A1F3C 50%, #0F1C3F 100%);
        background-attachment: fixed;
    }
    
    /* Animated background particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(46, 125, 158, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(124, 58, 237, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(236, 72, 153, 0.05) 0%, transparent 30%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Floating animation for background elements */
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-10px) rotate(1deg); }
        50% { transform: translateY(-20px) rotate(0deg); }
        75% { transform: translateY(-10px) rotate(-1deg); }
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
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            box-shadow: 0 0 20px rgba(46, 125, 158, 0.4),
                        0 0 40px rgba(46, 125, 158, 0.2),
                        0 0 60px rgba(46, 125, 158, 0.1);
        }
        50% { 
            box-shadow: 0 0 30px rgba(46, 125, 158, 0.6),
                        0 0 60px rgba(46, 125, 158, 0.4),
                        0 0 80px rgba(46, 125, 158, 0.2);
        }
    }
    
    @keyframes borderGlow {
        0%, 100% { border-color: rgba(46, 125, 158, 0.5); }
        50% { border-color: rgba(0, 212, 255, 0.8); }
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes successPop {
        0% { transform: scale(0.8); opacity: 0; }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes ripple {
        0% { transform: scale(0); opacity: 1; }
        100% { transform: scale(4); opacity: 0; }
    }
    
    /* ===================================================================
       MAIN HEADER STYLES
    =================================================================== */
    .main-header-container {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
        position: relative;
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00D4FF 0%, #2E7D9E 25%, #7C3AED 50%, #EC4899 75%, #00D4FF 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 8s linear infinite;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 40px rgba(46, 125, 158, 0.3);
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(46, 125, 158, 0.2), rgba(124, 58, 237, 0.2));
        border: 1px solid rgba(46, 125, 158, 0.3);
        border-radius: 50px;
        padding: 8px 20px;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: var(--accent-cyan);
        backdrop-filter: blur(10px);
        animation: fadeInUp 1s ease-out 0.3s backwards;
    }
    
    .header-badge::before {
        content: '';
        width: 8px;
        height: 8px;
        background: var(--success-green);
        border-radius: 50%;
        animation: pulse-glow 2s infinite;
    }
    
    /* ===================================================================
       GLASS MORPHISM CARDS
    =================================================================== */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    }
    
    .glass-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(46, 125, 158, 0.03) 0%, transparent 50%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(46, 125, 158, 0.3);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.3),
            0 0 40px rgba(46, 125, 158, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .glass-card:hover::after {
        opacity: 1;
    }
    
    /* ===================================================================
       FEATURE INPUT SECTION
    =================================================================== */
    .feature-section {
        background: linear-gradient(135deg, rgba(46, 125, 158, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(46, 125, 158, 0.2);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.7s ease-out;
    }
    
    .feature-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-purple));
        border-radius: 24px 24px 0 0;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .section-title-icon {
        font-size: 2rem;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .section-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        margin-bottom: 2rem;
        padding-left: 3rem;
    }
    
    /* ===================================================================
       INPUT FIELD STYLING
    =================================================================== */
    .input-group {
        background: rgba(15, 28, 63, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
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
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .input-group:hover {
        border-color: rgba(46, 125, 158, 0.3);
        background: rgba(15, 28, 63, 0.6);
        transform: translateX(5px);
    }
    
    .input-group:hover::before {
        opacity: 1;
    }
    
    .input-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .input-label-icon {
        font-size: 1.4rem;
    }
    
    .input-description {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
        padding-left: 2rem;
    }
    
    /* Streamlit Input Overrides */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid var(--accent-cyan) !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.5) !important;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(15, 28, 63, 0.8) !important;
        border: 2px solid rgba(46, 125, 158, 0.3) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        outline: none !important;
    }
    
    /* ===================================================================
       PREDICT BUTTON - ULTRA PREMIUM
    =================================================================== */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00D4FF 0%, #2E7D9E 25%, #7C3AED 50%, #EC4899 75%, #00D4FF 100%);
        background-size: 300% 300%;
        color: white !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        padding: 1.5rem 3rem !important;
        border-radius: 60px !important;
        border: none !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 
            0 10px 40px rgba(0, 212, 255, 0.4),
            0 0 80px rgba(124, 58, 237, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        animation: shimmer 4s linear infinite, pulse-glow 3s ease-in-out infinite !important;
        text-transform: uppercase !important;
        letter-spacing: 4px !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(
            120deg,
            transparent,
            rgba(255, 255, 255, 0.4),
            transparent
        ) !important;
        transition: left 0.8s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-8px) scale(1.02) !important;
        box-shadow: 
            0 20px 60px rgba(0, 212, 255, 0.5),
            0 0 100px rgba(124, 58, 237, 0.3),
            0 0 150px rgba(236, 72, 153, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        animation: shimmer 2s linear infinite !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(0.98) !important;
    }
    
    .stButton > button:focus {
        outline: none !important;
        border: none !important;
    }
    
    /* ===================================================================
       PREDICTION RESULTS
    =================================================================== */
    .result-container {
        animation: scaleIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .prediction-card {
        padding: 3rem;
        border-radius: 32px;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: successPop 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .prediction-stay {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(52, 211, 153, 0.1) 100%);
        border: 2px solid rgba(16, 185, 129, 0.4);
        box-shadow: 
            0 20px 60px rgba(16, 185, 129, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .prediction-stay::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--success-green), var(--success-light));
    }
    
    .prediction-leave {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(248, 113, 113, 0.1) 100%);
        border: 2px solid rgba(239, 68, 68, 0.4);
        box-shadow: 
            0 20px 60px rgba(239, 68, 68, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .prediction-leave::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--danger-red), var(--danger-light));
    }
    
    .prediction-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite;
    }
    
    .prediction-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: 2px;
    }
    
    .prediction-stay .prediction-title {
        background: linear-gradient(135deg, var(--success-green), var(--success-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .prediction-leave .prediction-title {
        background: linear-gradient(135deg, var(--danger-red), var(--danger-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .prediction-text {
        font-size: 1.3rem;
        color: var(--text-secondary);
        font-weight: 400;
    }
    
    /* ===================================================================
       PROBABILITY DISPLAY
    =================================================================== */
    .probability-container {
        background: linear-gradient(135deg, rgba(15, 28, 63, 0.8) 0%, rgba(26, 47, 90, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        animation: fadeInRight 0.8s ease-out 0.3s backwards;
    }
    
    .probability-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .probability-item {
        margin-bottom: 2rem;
    }
    
    .probability-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .probability-name {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1.1rem;
    }
    
    .probability-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        animation: countUp 0.8s ease-out;
    }
    
    .probability-value-stay {
        background: linear-gradient(135deg, var(--success-green), var(--success-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .probability-value-leave {
        background: linear-gradient(135deg, var(--danger-red), var(--danger-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .progress-bar-wrapper {
        width: 100%;
        height: 16px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 10px;
        position: relative;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    
    .progress-bar-stay {
        background: linear-gradient(90deg, var(--success-green), var(--success-light));
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
    }
    
    .progress-bar-leave {
        background: linear-gradient(90deg, var(--danger-red), var(--danger-light));
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
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
            rgba(255, 255, 255, 0.3),
            transparent
        );
        animation: shimmer 2s infinite;
    }
    
    /* ===================================================================
       SIDEBAR STYLING
    =================================================================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 28, 63, 0.95) 0%, rgba(10, 15, 28, 0.98) 100%) !important;
        border-right: 1px solid rgba(46, 125, 158, 0.2) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem !important;
    }
    
    .sidebar-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(46, 125, 158, 0.3);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .sidebar-card {
        background: linear-gradient(135deg, rgba(46, 125, 158, 0.15) 0%, rgba(124, 58, 237, 0.1) 100%);
        border: 1px solid rgba(46, 125, 158, 0.2);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .sidebar-card:hover {
        border-color: rgba(0, 212, 255, 0.4);
        transform: translateX(5px);
        box-shadow: 0 0 20px rgba(46, 125, 158, 0.2);
    }
    
    .sidebar-card-title {
        font-size: 0.85rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-card-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-cyan);
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .feature-list-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        background: rgba(15, 28, 63, 0.4);
        border-radius: 12px;
        border-left: 3px solid transparent;
        transition: all 0.3s ease;
        color: var(--text-secondary);
        font-size: 0.95rem;
    }
    
    .feature-list-item:hover {
        background: rgba(46, 125, 158, 0.15);
        border-left-color: var(--accent-cyan);
        color: var(--text-primary);
        transform: translateX(5px);
    }
    
    .feature-list-number {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
    }
    
    /* ===================================================================
       EXPANDER STYLING
    =================================================================== */
    div[data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }
    
    div[data-testid="stExpander"] details {
        border: 1px solid rgba(46, 125, 158, 0.3) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        background: linear-gradient(135deg, rgba(15, 28, 63, 0.6) 0%, rgba(26, 47, 90, 0.4) 100%) !important;
    }
    
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, rgba(46, 125, 158, 0.2) 0%, rgba(124, 58, 237, 0.15) 100%) !important;
        color: var(--text-primary) !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.5rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        background: linear-gradient(135deg, rgba(46, 125, 158, 0.3) 0%, rgba(124, 58, 237, 0.25) 100%) !important;
    }
    
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 16px 16px 0 0 !important;
    }
    
    div[data-testid="stExpander"] details > div {
        border-top: 1px solid rgba(46, 125, 158, 0.2) !important;
        padding: 1.5rem !important;
    }
    
    div[data-testid="stExpander"] details summary svg {
        color: var(--accent-cyan) !important;
    }
    
    /* ===================================================================
       TABLE STYLING
    =================================================================== */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .stDataFrame > div > div > div > div {
        background: rgba(15, 28, 63, 0.6) !important;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, rgba(46, 125, 158, 0.3) 0%, rgba(124, 58, 237, 0.2) 100%) !important;
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    .stDataFrame td {
        color: var(--text-secondary) !important;
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .stDataFrame tr:hover td {
        background: rgba(46, 125, 158, 0.1) !important;
    }
    
    /* ===================================================================
       DIVIDER STYLING
    =================================================================== */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(46, 125, 158, 0.5), rgba(124, 58, 237, 0.5), transparent);
        margin: 3rem 0;
        border: none;
    }
    
    hr {
        background: linear-gradient(90deg, transparent, rgba(46, 125, 158, 0.3), transparent) !important;
        border: none !important;
        height: 1px !important;
        margin: 2rem 0 !important;
    }
    
    /* ===================================================================
       INFO & ALERT BOXES
    =================================================================== */
    .info-box {
        background: linear-gradient(135deg, rgba(46, 125, 158, 0.15) 0%, rgba(0, 212, 255, 0.08) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        display: flex;
        gap: 15px;
        align-items: flex-start;
    }
    
    .info-box-icon {
        font-size: 1.5rem;
        color: var(--accent-cyan);
    }
    
    .info-box-content {
        flex: 1;
    }
    
    .info-box-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .info-box-text {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* ===================================================================
       METRIC DISPLAY CARDS
    =================================================================== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 28, 63, 0.8) 0%, rgba(26, 47, 90, 0.6) 100%);
        border: 1px solid rgba(46, 125, 158, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 212, 255, 0.4);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* ===================================================================
       FOOTER
    =================================================================== */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        margin-top: 4rem;
        border-top: 1px solid rgba(46, 125, 158, 0.2);
        color: var(--text-muted);
    }
    
    .footer-text {
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
    }
    
    .footer-link {
        color: var(--accent-cyan);
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer-link:hover {
        color: var(--accent-purple);
    }
    
    /* ===================================================================
       LOADING ANIMATION
    =================================================================== */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(46, 125, 158, 0.2);
        border-top-color: var(--accent-cyan);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* ===================================================================
       RESPONSIVE DESIGN
    =================================================================== */
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
        
        .feature-section {
            padding: 1.5rem;
        }
        
        .prediction-card {
            padding: 2rem;
        }
        
        .prediction-icon {
            font-size: 3rem;
        }
        
        .prediction-title {
            font-size: 2rem;
        }
        
        .stButton > button {
            font-size: 1.1rem !important;
            padding: 1rem 2rem !important;
        }
    }
    
    /* ===================================================================
       HIDE STREAMLIT BRANDING
    =================================================================== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===================================================================
       TOAST NOTIFICATIONS
    =================================================================== */
    .toast {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: linear-gradient(135deg, rgba(15, 28, 63, 0.95) 0%, rgba(26, 47, 90, 0.95) 100%);
        border: 1px solid rgba(46, 125, 158, 0.3);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        animation: slideInRight 0.5s ease-out;
        z-index: 9999;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
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

FEATURE_INFO = {
    "satisfaction_level": {
        "icon": "😊",
        "title": "Satisfaction Level",
        "description": "Employee's overall job satisfaction (0 = Very Dissatisfied, 1 = Very Satisfied)"
    },
    "time_spend_company": {
        "icon": "📅",
        "title": "Years at Company",
        "description": "Total tenure with the organization"
    },
    "average_monthly_hours": {
        "icon": "⏰",
        "title": "Monthly Hours",
        "description": "Average hours worked per month"
    },
    "number_project": {
        "icon": "📁",
        "title": "Active Projects",
        "description": "Number of concurrent projects assigned"
    },
    "last_evaluation": {
        "icon": "📊",
        "title": "Last Evaluation",
        "description": "Most recent performance evaluation score"
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
# HELPER FUNCTIONS
# ============================================================================
def create_animated_progress_bar(percentage, bar_type="stay"):
    """Create an animated progress bar with gradient"""
    bar_class = "progress-bar-stay" if bar_type == "stay" else "progress-bar-leave"
    return f"""
    <div class="progress-bar-wrapper">
        <div class="progress-bar-fill {bar_class}" style="width: {percentage}%;"></div>
    </div>
    """

def create_metric_card(icon, value, label):
    """Create a metric display card"""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    st.markdown("""
    <div class="main-header-container">
        <h1 class="main-header">👥 Employee Turnover Prediction</h1>
        <p class="sub-header">AI-Powered Workforce Analytics for Strategic Decision Making</p>
        <div class="header-badge">
            <span>Model Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model_from_huggingface()
    
    if model is None:
        st.markdown("""
        <div class="glass-card" style="border-color: rgba(239, 68, 68, 0.4);">
            <h3 style="color: #EF4444;">❌ Model Loading Failed</h3>
            <p style="color: var(--text-secondary);">Unable to connect to the model repository. Please check your connection and try again.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <span>🧠</span>
            <span>Model Information</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Model details cards
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">🤗 Repository</div>
            <div class="sidebar-card-value">{HF_REPO_ID}</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-card-title">🤖 Algorithm</div>
            <div class="sidebar-card-value">Random Forest Classifier</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-card-title">📊 Features</div>
            <div class="sidebar-card-value">{len(BEST_FEATURES)} Input Variables</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-card-title">⚖️ Class Balance</div>
            <div class="sidebar-card-value">Weighted</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-header">
            <span>📋</span>
            <span>Feature List</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature list with numbers
        feature_html = '<ul class="feature-list">'
        for i, feat in enumerate(BEST_FEATURES, 1):
            info = FEATURE_INFO.get(feat, {})
            icon = info.get("icon", "📌")
            title = info.get("title", feat.replace('_', ' ').title())
            feature_html += f"""
            <li class="feature-list-item">
                <span class="feature-list-number">{i}</span>
                <span>{icon} {title}</span>
            </li>
            """
        feature_html += '</ul>'
        st.markdown(feature_html, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-header">
            <span>🎯</span>
            <span>Prediction Classes</span>
        </div>
        <div class="info-box" style="margin-top: 0;">
            <div class="info-box-content">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="color: #10B981; font-size: 1.2rem;">✅</span>
                    <span style="color: var(--text-primary); font-weight: 600;">Stay</span>
                    <span style="color: var(--text-secondary);">- Likely to remain</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: #EF4444; font-size: 1.2rem;">⚠️</span>
                    <span style="color: var(--text-primary); font-weight: 600;">Leave</span>
                    <span style="color: var(--text-secondary);">- At risk of leaving</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: 2rem;">
            <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank" 
               style="color: var(--accent-cyan); text-decoration: none; 
                      display: inline-flex; align-items: center; gap: 8px;
                      padding: 10px 20px; border: 1px solid rgba(46, 125, 158, 0.3);
                      border-radius: 30px; transition: all 0.3s ease;">
                🔗 View on Hugging Face
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN INPUT SECTION
    # ========================================================================
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-section">
        <div class="section-title">
            <span class="section-title-icon">📝</span>
            <span>Employee Information</span>
        </div>
        <div class="section-subtitle">
            Enter the employee's metrics below to generate a turnover risk prediction
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'satisfaction_level' not in st.session_state:
        st.session_state.satisfaction_level = 0.5
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = 0.7
    
    # ========================================================================
    # INPUT FIELDS - Row 1
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">😊</span>
                <span>Satisfaction Level</span>
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
        
        st.markdown("""
        <div class="input-description">0 = Very Dissatisfied → 1 = Very Satisfied</div>
        """, unsafe_allow_html=True)
        
        satisfaction_level = st.session_state.satisfaction_level
    
    with col2:
        st.markdown("""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">📊</span>
                <span>Last Evaluation Score</span>
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
        
        st.markdown("""
        <div class="input-description">0 = Poor Performance → 1 = Excellent</div>
        """, unsafe_allow_html=True)
        
        last_evaluation = st.session_state.last_evaluation
    
    # ========================================================================
    # INPUT FIELDS - Row 2
    # ========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">📅</span>
                <span>Years at Company</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        time_spend_company = st.number_input(
            "Years",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            label_visibility="collapsed"
        )
        
        st.markdown("""
        <div class="input-description">Total years with the organization</div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">📁</span>
                <span>Number of Projects</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        number_project = st.number_input(
            "Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            label_visibility="collapsed"
        )
        
        st.markdown("""
        <div class="input-description">Currently assigned projects</div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="input-group">
            <div class="input-label">
                <span class="input-label-icon">⏰</span>
                <span>Average Monthly Hours</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        average_monthly_hours = st.number_input(
            "Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            label_visibility="collapsed"
        )
        
        st.markdown("""
        <div class="input-description">Typical hours worked per month</div>
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
    # PREDICT BUTTON
    # ========================================================================
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Generate Prediction", use_container_width=True)
    
    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================
    if predict_button:
        # Show loading animation
        with st.spinner(''):
            st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)  # Brief delay for effect
        
        # Create input DataFrame and make prediction
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_stay = prediction_proba[0] * 100
        prob_leave = prediction_proba[1] * 100
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-title" style="justify-content: center; margin-bottom: 2rem;">
            <span class="section-title-icon">🎯</span>
            <span>Prediction Results</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Results in two columns
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            if prediction == 0:
                st.markdown(f"""
                <div class="result-container">
                    <div class="prediction-card prediction-stay">
                        <div class="prediction-icon">✅</div>
                        <div class="prediction-title">STAY</div>
                        <p class="prediction-text">
                            This employee is predicted to <strong>remain</strong> with the company
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-container">
                    <div class="prediction-card prediction-leave">
                        <div class="prediction-icon">⚠️</div>
                        <div class="prediction-title">LEAVE</div>
                        <p class="prediction-text">
                            This employee is predicted to <strong>leave</strong> the company
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown(f"""
            <div class="probability-container">
                <div class="probability-title">
                    <span>📊</span>
                    <span>Prediction Confidence</span>
                </div>
                
                <div class="probability-item">
                    <div class="probability-label">
                        <span class="probability-name">✅ Probability of Staying</span>
                        <span class="probability-value probability-value-stay">{prob_stay:.1f}%</span>
                    </div>
                    {create_animated_progress_bar(prob_stay, "stay")}
                </div>
                
                <div class="probability-item">
                    <div class="probability-label">
                        <span class="probability-name">⚠️ Probability of Leaving</span>
                        <span class="probability-value probability-value-leave">{prob_leave:.1f}%</span>
                    </div>
                    {create_animated_progress_bar(prob_leave, "leave")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # RISK ANALYSIS & RECOMMENDATIONS
        # ====================================================================
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Determine risk level and recommendations
        if prob_leave >= 70:
            risk_level = "High Risk"
            risk_color = "#EF4444"
            risk_icon = "🔴"
            recommendations = [
                "Schedule an immediate one-on-one meeting",
                "Review compensation and benefits package",
                "Assess workload and project distribution",
                "Consider career development opportunities"
            ]
        elif prob_leave >= 40:
            risk_level = "Medium Risk"
            risk_color = "#F59E0B"
            risk_icon = "🟡"
            recommendations = [
                "Monitor engagement levels closely",
                "Provide additional recognition and feedback",
                "Discuss career growth opportunities",
                "Check in on work-life balance"
            ]
        else:
            risk_level = "Low Risk"
            risk_color = "#10B981"
            risk_icon = "🟢"
            recommendations = [
                "Maintain current engagement strategies",
                "Continue regular feedback sessions",
                "Recognize contributions and achievements",
                "Support professional development goals"
            ]
        
        st.markdown(f"""
        <div class="glass-card">
            <div class="section-title">
                <span>{risk_icon}</span>
                <span style="color: {risk_color};">{risk_level}</span>
            </div>
            <div class="info-box" style="margin-top: 1rem;">
                <div class="info-box-icon">💡</div>
                <div class="info-box-content">
                    <div class="info-box-title">Recommended Actions</div>
                    <ul style="color: var(--text-secondary); margin: 0; padding-left: 1.2rem;">
                        {"".join([f"<li style='margin-bottom: 0.5rem;'>{rec}</li>" for rec in recommendations])}
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ====================================================================
        # INPUT SUMMARY (Collapsible)
        # ====================================================================
        st.markdown("<br>", unsafe_allow_html=True)
        
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
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("""
    <div class="footer">
        <p class="footer-text">Powered by Machine Learning • Built with Streamlit</p>
        <div class="footer-links">
            <span style="color: var(--text-muted);">© 2024 Employee Turnover Prediction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
