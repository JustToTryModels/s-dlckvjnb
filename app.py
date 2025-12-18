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
# MODEL AND CONFIGURATION SETUP
# =============================

# Hugging Face model IDs
DistilGPT2_MODEL_ID = "IamPradeep/AETCSCB_OOD_IC_DistilGPT2_Fine-tuned"
CLASSIFIER_ID = "IamPradeep/Query_Classifier_DistilBERT"

# Random OOD Fallback Responses
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

# =============================
# PAGE CONFIGURATION
# =============================

st.set_page_config(
    page_title="Event Ticketing AI Assistant",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# ADVANCED CSS STYLING
# =============================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Animated Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main Container */
    .main {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding-bottom: 100px;
    }
    
    /* Animated Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-in;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Example Query Cards */
    .example-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .example-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .example-card:hover::before {
        left: 100%;
    }
    
    .example-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        border: 2px solid #fff;
    }
    
    .example-card-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    /* Chat Messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.5s ease;
        position: relative;
        max-width: 80%;
        margin-left: auto;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
        animation: slideInLeft 0.5s ease;
        position: relative;
        max-width: 80%;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Timestamp */
    .timestamp {
        font-size: 0.7rem;
        opacity: 0.8;
        margin-top: 0.5rem;
        text-align: right;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
        margin: 0.5rem 0;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    .stButton>button:disabled {
        background: linear-gradient(135deg, #ccc 0%, #999 100%);
        cursor: not-allowed;
        transform: none;
    }
    
    /* Chat Input */
    .stChatInput {
        border-radius: 25px;
        border: 2px solid #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stChatInput:focus {
        border: 2px solid #764ba2;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    section[data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Statistics Cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .stat-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        padding: 1rem;
    }
    
    .typing-indicator span {
        height: 10px;
        width: 10px;
        background: white;
        border-radius: 50%;
        display: inline-block;
        margin: 0 3px;
        animation: typing 1.4s infinite;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    /* Loading Spinner */
    .loader {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 4px solid white;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Progress Bar */
    .progress-bar {
        width: 100%;
        height: 4px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 2px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        animation: progress 2s ease;
    }
    
    @keyframes progress {
        from { width: 0%; }
        to { width: 100%; }
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 1rem 0;
        font-size: 0.9rem;
        z-index: 9999;
        box-shadow: 0 -4px 15px rgba(0, 0, 0, 0.2);
        animation: slideInUp 1s ease;
    }
    
    @keyframes slideInUp {
        from {
            transform: translateY(100%);
        }
        to {
            transform: translateY(0);
        }
    }
    
    /* Welcome Message */
    .welcome-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
    }
    
    /* Action Buttons */
    .action-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid white;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .action-button:hover {
        background: white;
        color: #667eea;
        transform: scale(1.05);
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* Quick Suggestions */
    .quick-suggestion {
        display: inline-block;
        background: rgba(102, 126, 234, 0.1);
        border: 2px solid #667eea;
        color: #667eea;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .quick-suggestion:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Notification Toast */
    .stToast {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 15px;
        border: 2px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border: 2px solid #764ba2;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Copy Button */
    .copy-btn {
        background: rgba(255, 255, 255, 0.3);
        border: 1px solid white;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 10px;
        cursor: pointer;
        font-size: 0.8rem;
        transition: all 0.3s ease;
        margin-top: 0.5rem;
        display: inline-block;
    }
    
    .copy-btn:hover {
        background: white;
        color: #667eea;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE INITIALIZATION
# =============================

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "generating" not in st.session_state:
    st.session_state.generating = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "message_count" not in st.session_state:
    st.session_state.message_count = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

# =============================
# SIDEBAR WITH STATISTICS
# =============================

with st.sidebar:
    st.markdown("<h2 style='color: white; text-align: center;'>📊 Session Stats</h2>", unsafe_allow_html=True)
    
    # Calculate session duration
    if st.session_state.session_start:
        duration = datetime.now() - st.session_state.session_start
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = "0m 0s"
    
    # Statistics Cards
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(st.session_state.chat_history) // 2}</div>
            <div class="stat-label">💬 Messages Exchanged</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{duration_str}</div>
            <div class="stat-label">⏱️ Session Duration</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{'🟢 Online' if st.session_state.models_loaded else '🔴 Loading'}</div>
            <div class="stat-label">🤖 Bot Status</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("<h3 style='color: white;'>⚡ Quick Actions</h3>", unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Chat History", key="sidebar_clear", disabled=st.session_state.generating):
        st.session_state.chat_history = []
        st.session_state.message_count = 0
        st.session_state.session_start = datetime.now()
        st.session_state.show_welcome = True
        st.rerun()
    
    if st.button("🔄 Restart Session", key="sidebar_restart", disabled=st.session_state.generating):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; color: white;'>
            <h4 style='margin: 0;'>ℹ️ About</h4>
            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
                This AI assistant is specialized in handling event ticketing queries. 
                Ask anything about bookings, cancellations, refunds, and more!
            </p>
        </div>
    """, unsafe_allow_html=True)

# =============================
# MAIN CONTENT AREA
# =============================

# Header
st.markdown("<h1 class='main-header'>🎫 Event Ticketing AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your intelligent companion for all event ticketing needs ✨</p>", unsafe_allow_html=True)

# Example queries with categories
example_queries = {
    "🎟️ Booking": [
        "How do I buy a ticket?",
        "How can I find details about upcoming events?"
    ],
    "⬆️ Upgrades": [
        "How can I upgrade my ticket for the upcoming event in Hyderabad?",
        "How do I change my personal details on my ticket?"
    ],
    "❌ Cancellations": [
        "What is the ticket cancellation fee?",
        "How can I track my ticket cancellation status?"
    ],
    "💰 Refunds": [
        "How do I get a refund?",
        "How can I sell my ticket?"
    ],
    "📞 Support": [
        "How do I contact customer service?"
    ]
}

# =============================
# MODEL LOADING
# =============================

if not st.session_state.models_loaded:
    st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <div class='loader'></div>
            <h3 style='color: #667eea; margin-top: 1rem;'>Loading AI Models...</h3>
            <p style='color: #666;'>Please wait while we initialize the chatbot</p>
            <div class='progress-bar'>
                <div class='progress-bar-fill'></div>
            </div>
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
            st.success("✅ Models loaded successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Failed to load one or more models. Please refresh the page.")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")

# =============================
# MAIN CHAT INTERFACE
# =============================

if st.session_state.models_loaded:
    
    # Welcome Message
    if st.session_state.show_welcome and len(st.session_state.chat_history) == 0:
        st.markdown("""
            <div class='welcome-message'>
                <h2 style='margin: 0;'>👋 Welcome to Event Ticketing Assistant!</h2>
                <p style='margin-top: 1rem; font-size: 1.1rem;'>
                    I'm here to help you with all your event ticketing needs. 
                    Choose a question below or type your own!
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Example Query Cards by Category
    if len(st.session_state.chat_history) == 0:
        st.markdown("<h3 style='text-align: center; color: #667eea; margin: 2rem 0 1rem 0;'>💡 Try These Questions</h3>", unsafe_allow_html=True)
        
        for category, queries in example_queries.items():
            with st.expander(f"**{category}**", expanded=False):
                for query in queries:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"<div style='padding: 0.5rem;'>• {query}</div>", unsafe_allow_html=True)
                    with col2:
                        if st.button("Ask", key=f"btn_{query}", disabled=st.session_state.generating):
                            st.session_state.selected_query = query
                            st.rerun()
    
    # Process selected query from buttons
    if hasattr(st.session_state, 'selected_query') and st.session_state.selected_query:
        query = st.session_state.selected_query
        st.session_state.selected_query = None
        st.session_state.generating = True
        query = query[0].upper() + query[1:]
        st.session_state.chat_history.append({
            "role": "user", 
            "content": query, 
            "avatar": "👤",
            "timestamp": datetime.now().strftime("%H:%M")
        })
        st.session_state.show_welcome = False
        st.rerun()
    
    # Divider
    if len(st.session_state.chat_history) > 0:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Display Chat History
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer
    
    for idx, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(f"""
                <div style='font-size: 1.05rem;'>{message["content"]}</div>
                <div class='timestamp'>{message.get('timestamp', '')}</div>
            """, unsafe_allow_html=True)
            
            # Add copy button for assistant messages
            if message["role"] == "assistant":
                if st.button("📋 Copy Response", key=f"copy_{idx}"):
                    st.toast("✅ Response copied to clipboard!", icon="✅")
    
    # Typing Indicator when generating
    if st.session_state.generating and len(st.session_state.chat_history) > 0:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("""
                <div class='typing-indicator'>
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            """, unsafe_allow_html=True)
    
    # Chat Input
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    with col1:
        prompt = st.chat_input(
            "💭 Type your question here...", 
            disabled=st.session_state.generating,
            key="chat_input"
        )
    with col2:
        if st.button("🗑️", key="clear_main", disabled=st.session_state.generating, help="Clear chat"):
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            st.session_state.show_welcome = True
            st.rerun()
    
    # Handle user input
    def handle_prompt(prompt_text):
        if not prompt_text or not prompt_text.strip():
            st.toast("⚠️ Please enter a question.", icon="⚠️")
            return
        
        st.session_state.generating = True
        prompt_text = prompt_text[0].upper() + prompt_text[1:]
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt_text, 
            "avatar": "👤",
            "timestamp": datetime.now().strftime("%H:%M")
        })
        st.session_state.show_welcome = False
        st.rerun()
    
    def process_generation():
        last_message = st.session_state.chat_history[-1]["content"]
        
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Show typing indicator
            message_placeholder.markdown("""
                <div class='typing-indicator'>
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            """, unsafe_allow_html=True)
            
            if is_ood(last_message, clf_model, clf_tokenizer):
                full_response = random.choice(fallback_responses)
            else:
                dynamic_placeholders = extract_dynamic_placeholders(last_message, nlp)
                response_gpt = generate_response(model, tokenizer, last_message)
                full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
            
            # Animated typing effect
            streamed_text = ""
            for word in full_response.split(" "):
                streamed_text += word + " "
                message_placeholder.markdown(f"""
                    <div style='font-size: 1.05rem;'>{streamed_text}▊</div>
                """, unsafe_allow_html=True)
                time.sleep(0.03)
            
            # Final message with timestamp
            timestamp = datetime.now().strftime("%H:%M")
            message_placeholder.markdown(f"""
                <div style='font-size: 1.05rem;'>{full_response}</div>
                <div class='timestamp'>{timestamp}</div>
            """, unsafe_allow_html=True)
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": full_response, 
            "avatar": "🤖",
            "timestamp": timestamp
        })
        st.session_state.generating = False
        st.session_state.message_count += 1
    
    # Process prompt
    if prompt:
        handle_prompt(prompt)
    
    # Generate response if needed
    if st.session_state.generating and len(st.session_state.chat_history) > 0:
        if st.session_state.chat_history[-1]["role"] == "user":
            process_generation()
            st.rerun()

# =============================
# FOOTER
# =============================

st.markdown("""
    <div class="footer">
        <strong>⚠️ Disclaimer:</strong> This is not a conversational AI. It is designed solely for <b>event ticketing</b> queries. 
        Responses outside this scope may be inaccurate. | 💡 Powered by AI Technology
    </div>
""", unsafe_allow_html=True)
