import streamlit as st
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSequenceClassification
)
import spacy
import time
import random
from datetime import datetime

# =============================
# PAGE CONFIGURATION
# =============================

st.set_page_config(
    page_title="🎫 Event Ticketing Chatbot",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# MODEL AND CONFIGURATION SETUP
# =============================

DistilGPT2_MODEL_ID = "IamPradeep/AETCSCB_OOD_IC_DistilGPT2_Fine-tuned"
CLASSIFIER_ID = "IamPradeep/Query_Classifier_DistilBERT"

fallback_responses = [
    "I'm sorry, but I am unable to assist with this request. If you need help regarding event tickets, I'd be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    "Unfortunately, I cannot assist with this. However, I am here to help with any event ticket-related concerns you may have.",
    "Regrettably, I am unable to assist with this request. If there's anything I can do regarding event tickets, feel free to ask.",
    "I regret that I am unable to assist in this case. Please reach out if you need support related to event tickets.",
    "Apologies, but this falls outside the scope of my support. I'm here if you need any help with event ticket issues.",
    "I'm sorry, but I cannot assist with this particular topic. If you have questions about event tickets, I'd be glad to help.",
    "I regret that I'm unable to provide assistance here. Please let me know how I can support you with event ticket matters.",
    "Unfortunately, I am not equipped to assist with this. If you need help with event tickets, I am here for that.",
    "I apologize, but I cannot help with this request. However, I'd be happy to assist with anything related to event tickets.",
    "I'm sorry, but I'm unable to support this request. If it's about event tickets, I'll gladly help however I can.",
    "This matter falls outside the assistance I can offer. Please let me know if you need help with event ticket-related inquiries.",
    "Regrettably, this is not something I can assist with. I'm happy to help with any event ticket questions you may have.",
    "I'm unable to provide support for this issue. However, I can assist with concerns regarding event tickets.",
    "I apologize, but I cannot help with this matter. If your inquiry is related to event tickets, I'd be more than happy to assist.",
    "I regret that I am unable to offer help in this case. I am, however, available for any event ticket-related questions.",
    "Unfortunately, I'm not able to assist with this. Please let me know if there's anything I can do regarding event tickets.",
    "I'm sorry, but I cannot assist with this topic. However, I'm here to help with any event ticket concerns you may have.",
    "Apologies, but this request falls outside of my support scope. If you need help with event tickets, I'm happy to assist.",
    "I'm afraid I can't help with this matter. If there's anything related to event tickets you need, feel free to reach out.",
    "This is beyond what I can assist with at the moment. Let me know if there's anything I can do to help with event tickets.",
    "Sorry, I'm unable to provide support on this issue. However, I'd be glad to assist with event ticket-related topics.",
    "Apologies, but I can't assist with this. Please let me know if you have any event ticket inquiries I can help with.",
    "I'm unable to help with this matter. However, if you need assistance with event tickets, I'm here for you.",
    "Unfortunately, I can't support this request. I'd be happy to assist with anything related to event tickets instead.",
    "I'm sorry, but I can't help with this. If your concern is related to event tickets, I'll do my best to assist.",
    "Apologies, but this issue is outside of my capabilities. However, I'm available to help with event ticket-related requests.",
    "I regret that I cannot assist with this particular matter. Please let me know how I can support you regarding event tickets.",
    "I'm sorry, but I'm not able to help in this instance. I am, however, ready to assist with any questions about event tickets.",
    "Unfortunately, I'm unable to help with this topic. Let me know if there's anything event ticket-related I can support you with."
]

# =============================
# ENHANCED CSS STYLING
# =============================

st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Poppins', sans-serif !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 1200px;
}

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #0f0f23, #1a1a2e, #16213e, #0f0f23);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Animated Header */
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(
        45deg,
        transparent,
        rgba(102, 126, 234, 0.1),
        transparent
    );
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) rotate(45deg); }
    100% { transform: translateX(100%) rotate(45deg); }
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    animation: titleGlow 2s ease-in-out infinite alternate;
    position: relative;
    z-index: 1;
}

@keyframes titleGlow {
    from { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.5)); }
    to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.8)); }
}

.main-subtitle {
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.7);
    margin-top: 0.5rem;
    position: relative;
    z-index: 1;
}

.emoji-float {
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

/* Status indicator */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: rgba(46, 213, 115, 0.15);
    border: 1px solid rgba(46, 213, 115, 0.3);
    border-radius: 20px;
    margin-top: 1rem;
    position: relative;
    z-index: 1;
}

.status-dot {
    width: 10px;
    height: 10px;
    background: #2ed573;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

.status-text {
    color: #2ed573;
    font-size: 0.85rem;
    font-weight: 500;
}

/* Quick action chips */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 1rem 0;
    justify-content: center;
}

.query-chip {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 10px 18px;
    border-radius: 25px;
    color: white;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(5px);
    position: relative;
    overflow: hidden;
}

.query-chip:hover {
    transform: translateY(-3px) scale(1.02);
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.4), rgba(118, 75, 162, 0.4));
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    border-color: rgba(255, 255, 255, 0.3);
}

.query-chip::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.query-chip:hover::before {
    left: 100%;
}

/* Chat container */
.chat-container {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    min-height: 400px;
    max-height: 600px;
    overflow-y: auto;
}

/* Chat messages */
.user-message {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 15px 20px;
    border-radius: 20px 20px 5px 20px;
    margin: 10px 0;
    max-width: 80%;
    margin-left: auto;
    animation: slideInRight 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    position: relative;
}

.bot-message {
    background: rgba(255, 255, 255, 0.08);
    color: white;
    padding: 15px 20px;
    border-radius: 20px 20px 20px 5px;
    margin: 10px 0;
    max-width: 80%;
    animation: slideInLeft 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}

.message-time {
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.5);
    margin-top: 8px;
    text-align: right;
}

.message-avatar {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 5px;
}

.user-avatar {
    background: linear-gradient(135deg, #667eea, #764ba2);
    margin-left: auto;
}

.bot-avatar {
    background: linear-gradient(135deg, #f093fb, #f5576c);
}

/* Typing indicator */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 15px 20px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    width: fit-content;
    margin: 10px 0;
    animation: fadeIn 0.3s ease;
}

.typing-dot {
    width: 8px;
    height: 8px;
    background: rgba(255, 255, 255, 0.6);
    border-radius: 50%;
    animation: typingBounce 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) { animation-delay: 0s; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Enhanced buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none;
    border-radius: 15px;
    padding: 12px 28px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
}

.stButton > button:active {
    transform: translateY(0);
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: left 0.5s;
}

.stButton > button:hover::before {
    left: 100%;
}

/* Clear button special styling */
.clear-btn button {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
}

/* Chat input styling */
div[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 5px;
    transition: all 0.3s ease;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: rgba(102, 126, 234, 0.5);
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
}

div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: white !important;
}

/* Selectbox styling */
div[data-testid="stSelectbox"] > div {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 15, 35, 0.95), rgba(26, 26, 46, 0.95));
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.sidebar-header {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
    border-radius: 15px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.sidebar-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: white;
    margin-bottom: 0.5rem;
}

.sidebar-subtitle {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.6);
}

/* Feature cards */
.feature-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.8rem 0;
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: all 0.3s ease;
}

.feature-card:hover {
    background: rgba(255, 255, 255, 0.08);
    transform: translateX(5px);
    border-color: rgba(102, 126, 234, 0.3);
}

.feature-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.feature-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: white;
    margin-bottom: 0.3rem;
}

.feature-desc {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.6);
}

/* Stats cards */
.stats-container {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin: 1rem 0;
}

.stat-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: scale(1.05);
}

.stat-number {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.6);
    margin-top: 0.3rem;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    margin: 1.5rem 0;
}

/* Info box */
.info-box {
    background: linear-gradient(135deg, rgba(46, 213, 115, 0.1), rgba(39, 174, 96, 0.1));
    border: 1px solid rgba(46, 213, 115, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}

.info-box-title {
    color: #2ed573;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.info-box-content {
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.85rem;
    line-height: 1.6;
}

/* Warning box */
.warning-box {
    background: linear-gradient(135deg, rgba(255, 165, 2, 0.1), rgba(255, 107, 107, 0.1));
    border: 1px solid rgba(255, 165, 2, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box-title {
    color: #ffa502;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Footer */
.custom-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: linear-gradient(180deg, transparent, rgba(15, 15, 35, 0.95));
    padding: 15px 0;
    text-align: center;
    z-index: 9999;
}

.footer-content {
    background: rgba(255, 255, 255, 0.05);
    display: inline-block;
    padding: 10px 25px;
    border-radius: 25px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}

.footer-text {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.85rem;
}

.footer-highlight {
    color: #667eea;
    font-weight: 600;
}

/* Horizontal line between messages */
.message-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    margin: 20px 0;
}

/* Loading spinner */
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
    border: 4px solid rgba(255, 255, 255, 0.1);
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-text {
    color: rgba(255, 255, 255, 0.7);
    font-size: 1rem;
    animation: pulse 1.5s ease-in-out infinite;
}

/* Quick actions section */
.quick-actions-title {
    font-size: 1rem;
    color: rgba(255, 255, 255, 0.8);
    margin-bottom: 1rem;
    text-align: center;
    font-weight: 500;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2, #667eea);
}

/* Tooltip styling */
.tooltip {
    position: relative;
    display: inline-block;
}

.tooltip .tooltiptext {
    visibility: hidden;
    background: rgba(0, 0, 0, 0.9);
    color: #fff;
    text-align: center;
    padding: 8px 12px;
    border-radius: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.8rem;
    white-space: nowrap;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

/* Glow effect for important elements */
.glow {
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
}

/* Notification badge */
.notification-badge {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: 8px;
}

/* Category pills */
.category-pill {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(102, 126, 234, 0.2);
    border-radius: 15px;
    font-size: 0.75rem;
    color: #667eea;
    margin: 2px;
}

/* Animated border */
.animated-border {
    position: relative;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
}

.animated-border::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #667eea);
    background-size: 400% 400%;
    border-radius: 17px;
    z-index: -1;
    animation: borderGlow 3s ease infinite;
}

@keyframes borderGlow {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Celebration effect */
@keyframes celebrate {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.celebrate {
    animation: celebrate 0.5s ease;
}
</style>
""", unsafe_allow_html=True)

# =============================
# MODEL LOADING FUNCTIONS
# =============================

@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

@st.cache_resource(show_spinner=False)
def load_gpt2_model_and_tokenizer():
    try:
        model = GPT2LMHeadModel.from_pretrained(DistilGPT2_MODEL_ID, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(DistilGPT2_MODEL_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load GPT-2 model from Hugging Face Hub. Error: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_classifier_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load classifier model from Hugging Face Hub. Error: {e}")
        return None, None

def is_ood(query: str, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return pred_id == 1

# =============================
# HELPER FUNCTIONS
# =============================

static_placeholders = {
    "{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_TEAM_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CONTACT_SUPPORT_LINK}}" : "[support team](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_CONTACT_LINK}}" : "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CANCEL_TICKET_SECTION}}": "<b>Ticket Cancellation</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Upgrade Ticket Information</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>",
    "{{APP}}": "<b>App</b>",
    "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>",
    "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>",
    "{{SAVE_BUTTON}}": "<b>Save</b>",
    "{{EDIT_BUTTON}}": "<b>Edit</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>",
    "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>",
    "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>",
    "{{REFUND_SECTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>",
    "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>",
    "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>",
    "{{CONTACT_SECTION}}": "<b>Contact</b>",
    "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SUPPORT_SECTION}}": "<b>Support</b>",
    "{{EVENTS_SECTION}}": "<b>Events</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>",
    "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>",
    "{{PAYMENT_SECTION}}": "<b>Payment</b>",
    "{{PAYMENT_OPTION}}": "<b>Payment</b>",
    "{{CANCELLATION_SECTION}}": "<b>Cancellation</b>",
    "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>",
    "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>",
    "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>",
    "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{PAYMENT_METHOD}}" : "<b>Payment</b>",
    "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{SUPPORT_ SECTION}}" : "<b>Support</b>",
    "{{CUSTOMER_SUPPORT_SECTION}}" : "<b>Customer Support</b>",
    "{{HELP_SECTION}}" : "<b>Help</b>",
    "{{TICKET_INFORMATION}}" : "<b>Ticket Information</b>",
    "{{UPGRADE_TICKET_BUTTON}}" : "<b>Upgrade Ticket</b>",
    "{{CANCEL_TICKET_BUTTON}}" : "<b>Cancel Ticket</b>",
    "{{GET_REFUND_BUTTON}}" : "<b>Get Refund</b>",
    "{{PAYMENTS_HELP_SECTION}}" : "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}" : "<b>Payments</b>",
    "{{TICKET_DETAILS}}" : "<b>Ticket Details</b>",
    "{{TICKET_INFORMATION_PAGE}}" : "<b>Ticket Information</b>",
    "{{REPORT_PAYMENT_PROBLEM}}" : "<b>Report Payment</b>",
    "{{TICKET_OPTIONS}}" : "<b>Ticket Options</b>",
    "{{SEND_BUTTON}}" : "<b>Send</b>",
    "{{PAYMENT_ISSUE_OPTION}}" : "<b>Payment Issue</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}" : "<b>Customer Support</b>",
    "{{UPGRADE_TICKET_OPTION}}" : "<b>Upgrade Ticket</b>",
    "{{TICKET_AVAILABILITY_TAB}}" : "<b>Ticket Availability</b>",
    "{{TRANSFER_TICKET_BUTTON}}" : "<b>Transfer Ticket</b>",
    "{{TICKET_MANAGEMENT}}" : "<b>Ticket Management</b>",
    "{{TICKET_STATUS_TAB}}" : "<b>Ticket Status</b>",
    "{{TICKETING_PAGE}}" : "<b>Ticketing</b>",
    "{{TICKET_TRANSFER_TAB}}" : "<b>Ticket Transfer</b>",
    "{{CURRENT_TICKET_DETAILS}}" : "<b>Current Ticket Details</b>",
    "{{UPGRADE_OPTION}}" : "<b>Upgrade</b>",
    "{{CONNECT_WITH_ORGANIZER}}" : "<b>Connect with Organizer</b>",
    "{{TICKETS_TAB}}" : "<b>Tickets</b>",
    "{{ASSISTANCE_SECTION}}" : "<b>Assistance Section</b>",
}

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

def extract_dynamic_placeholders(user_question, nlp):
    doc = nlp(user_question)
    dynamic_placeholders = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    return dynamic_placeholders

def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.4,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

def get_current_time():
    return datetime.now().strftime("%I:%M %p")

# =============================
# INITIALIZE SESSION STATE
# =============================

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "generating" not in st.session_state:
    st.session_state.generating = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_chip" not in st.session_state:
    st.session_state.selected_chip = None
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

example_queries = [
    "🎫 How do I buy a ticket?",
    "⬆️ How can I upgrade my ticket?",
    "✏️ How do I change my ticket details?",
    "📅 Find details about upcoming events",
    "📞 How do I contact customer service?",
    "💰 How do I get a refund?",
    "❌ What is the ticket cancellation fee?",
    "🔍 Track my ticket cancellation status",
    "💵 How can I sell my ticket?"
]

example_queries_clean = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?",
    "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation status?",
    "How can I sell my ticket?"
]

# =============================
# SIDEBAR
# =============================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎫</div>
        <div class="sidebar-title">Event Ticketing Assistant</div>
        <div class="sidebar-subtitle">Your AI-powered ticket helper</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats
    st.markdown("""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">{}</div>
            <div class="stat-label">Messages</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{}</div>
            <div class="stat-label">Queries</div>
        </div>
    </div>
    """.format(
        len(st.session_state.chat_history),
        len([m for m in st.session_state.chat_history if m["role"] == "user"])
    ), unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Features
    st.markdown("### ✨ Features")
    
    features = [
        ("🎟️", "Ticket Booking", "Purchase tickets for events"),
        ("🔄", "Refunds & Cancellations", "Process refunds easily"),
        ("📍", "Event Discovery", "Find upcoming events"),
        ("💳", "Payment Support", "Secure payment assistance"),
        ("📧", "Customer Service", "Get help when needed")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">💡 Pro Tip</div>
        <div class="info-box-content">
            Click on the quick action buttons in the main chat area for common questions, or type your own custom query!
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Warning box
    st.markdown("""
    <div class="warning-box">
        <div class="warning-box-title">⚠️ Note</div>
        <div class="info-box-content">
            This chatbot is specialized for event ticketing queries only. Other topics may not receive accurate responses.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================
# MAIN CONTENT
# =============================

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">
        <span class="emoji-float">🎫</span> Advanced Event Ticketing Chatbot
    </div>
    <div class="main-subtitle">
        Your intelligent assistant for all event ticketing needs
    </div>
    <div class="status-indicator">
        <div class="status-dot"></div>
        <span class="status-text">AI Assistant Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Model loading
if not st.session_state.models_loaded:
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">Initializing AI models... Please wait...</div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        nlp = load_spacy_model()
        gpt2_model, gpt2_tokenizer = load_gpt2_model_and_tokenizer()
        clf_model, clf_tokenizer = load_classifier_model()

        if all([nlp, gpt2_model, gpt2_tokenizer, clf_model, clf_tokenizer]):
            st.session_state.models_loaded = True
            st.session_state.nlp = nlp
            st.session_state.model = gpt2_model
            st.session_state.tokenizer = gpt2_tokenizer
            st.session_state.clf_model = clf_model
            st.session_state.clf_tokenizer = clf_tokenizer
            st.rerun()
        else:
            st.error("❌ Failed to load one or more models. Please refresh the page.")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")

# Main chat interface
if st.session_state.models_loaded:
    
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer
    
    # Quick actions section
    st.markdown('<p class="quick-actions-title">⚡ Quick Actions - Click to ask</p>', unsafe_allow_html=True)
    
    # Create chip buttons in columns
    cols = st.columns(3)
    for idx, (query, clean_query) in enumerate(zip(example_queries, example_queries_clean)):
        col_idx = idx % 3
        with cols[col_idx]:
            if st.button(query, key=f"chip_{idx}", disabled=st.session_state.generating, use_container_width=True):
                st.session_state.selected_chip = clean_query
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            if i > 0:
                st.markdown('<div class="message-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: flex-end;">
                <div class="message-avatar user-avatar">👤</div>
                <div class="user-message">
                    {message["content"]}
                    <div class="message-time">{message.get("time", "")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: flex-start;">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="bot-message">
                    {message["content"]}
                    <div class="message-time">{message.get("time", "")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Handle prompt
    def handle_prompt(prompt_text):
        if not prompt_text or not prompt_text.strip():
            st.toast("⚠️ Please enter or select a question.", icon="⚠️")
            return
        
        st.session_state.generating = True
        prompt_text = prompt_text[0].upper() + prompt_text[1:]
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt_text,
            "avatar": "👤",
            "time": get_current_time()
        })
        st.session_state.message_count += 1
        st.rerun()

    def process_generation():
        last_message = st.session_state.chat_history[-1]["content"]
        
        # Show typing indicator
        st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        """, unsafe_allow_html=True)
        
        if is_ood(last_message, clf_model, clf_tokenizer):
            full_response = random.choice(fallback_responses)
        else:
            dynamic_placeholders = extract_dynamic_placeholders(last_message, nlp)
            response_gpt = generate_response(model, tokenizer, last_message)
            full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
        
        # Simulate streaming with typing effect
        message_placeholder = st.empty()
        streamed_text = ""
        for word in full_response.split(" "):
            streamed_text += word + " "
            message_placeholder.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: flex-start;">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="bot-message">
                    {streamed_text}⬤
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.03)
        
        message_placeholder.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: flex-start;">
            <div class="message-avatar bot-avatar">🤖</div>
            <div class="bot-message celebrate">
                {full_response}
                <div class="message-time">{get_current_time()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "avatar": "🤖",
            "time": get_current_time()
        })
        st.session_state.generating = False

    # Process chip selection
    if st.session_state.selected_chip:
        handle_prompt(st.session_state.selected_chip)
        st.session_state.selected_chip = None
    
    # Chat input
    if prompt := st.chat_input("✨ Type your question here...", disabled=st.session_state.generating):
        handle_prompt(prompt)
    
    # Process generation if needed
    if st.session_state.generating:
        process_generation()
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Conversation", key="reset_button", disabled=st.session_state.generating, use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.generating = False
                st.session_state.message_count = 0
                st.balloons()
                st.rerun()

# Footer
st.markdown("""
<div class="custom-footer">
    <div class="footer-content">
        <span class="footer-text">
            🎫 Specialized for <span class="footer-highlight">Event Ticketing</span> queries • 
            Built with ❤️ using Streamlit & Transformers
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
