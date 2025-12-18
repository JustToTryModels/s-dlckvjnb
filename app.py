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
    "I’m sorry, but I am unable to assist with this request. If you need help regarding event tickets, I’d be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    "Unfortunately, I cannot assist with this. However, I am here to help with any event ticket-related concerns you may have.",
    "Regrettably, I am unable to assist with this request. If there's anything I can do regarding event tickets, feel free to ask.",
    "I regret that I am unable to assist in this case. Please reach out if you need support related to event tickets.",
    "Apologies, but this falls outside the scope of my support. I’m here if you need any help with event ticket issues.",
    "I'm sorry, but I cannot assist with this particular topic. If you have questions about event tickets, I’d be glad to help.",
    "I regret that I’m unable to provide assistance here. Please let me know how I can support you with event ticket matters.",
    "Unfortunately, I am not equipped to assist with this. If you need help with event tickets, I am here for that.",
    "I apologize, but I cannot help with this request. However, I’d be happy to assist with anything related to event tickets.",
    "I’m sorry, but I’m unable to support this request. If it’s about event tickets, I’ll gladly help however I can.",
    "This matter falls outside the assistance I can offer. Please let me know if you need help with event ticket-related inquiries.",
    "Regrettably, this is not something I can assist with. I’m happy to help with any event ticket questions you may have.",
    "I’m unable to provide support for this issue. However, I can assist with concerns regarding event tickets.",
    "I apologize, but I cannot help with this matter. If your inquiry is related to event tickets, I’d be more than happy to assist.",
    "I regret that I am unable to offer help in this case. I am, however, available for any event ticket-related questions.",
    "Unfortunately, I’m not able to assist with this. Please let me know if there’s anything I can do regarding event tickets.",
    "I'm sorry, but I cannot assist with this topic. However, I’m here to help with any event ticket concerns you may have.",
    "Apologies, but this request falls outside of my support scope. If you need help with event tickets, I’m happy to assist.",
    "I’m afraid I can’t help with this matter. If there’s anything related to event tickets you need, feel free to reach out.",
    "This is beyond what I can assist with at the moment. Let me know if there’s anything I can do to help with event tickets.",
    "Sorry, I’m unable to provide support on this issue. However, I’d be glad to assist with event ticket-related topics.",
    "Apologies, but I can’t assist with this. Please let me know if you have any event ticket inquiries I can help with.",
    "I’m unable to help with this matter. However, if you need assistance with event tickets, I’m here for you.",
    "Unfortunately, I can’t support this request. I’d be happy to assist with anything related to event tickets instead.",
    "I’m sorry, but I can’t help with this. If your concern is related to event tickets, I’ll do my best to assist.",
    "Apologies, but this issue is outside of my capabilities. However, I’m available to help with event ticket-related requests.",
    "I regret that I cannot assist with this particular matter. Please let me know how I can support you regarding event tickets.",
    "I’m sorry, but I’m not able to help in this instance. I am, however, ready to assist with any questions about event tickets.",
    "Unfortunately, I’m unable to help with this topic. Let me know if there's anything event ticket-related I can support you with."
]

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
    return pred_id == 1  # True if OOD (label 1)

# =============================
# HELPER FUNCTIONS
# =============================

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
# ENHANCED CSS STYLING
# =============================

def apply_advanced_styling():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e7f4 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 1rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: slideDown 0.5s ease-out;
    }
    
    .app-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #fff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .app-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
        height: calc(100vh - 300px);
        overflow-y: auto;
        scroll-behavior: smooth;
    }
    
    /* Chat bubbles */
    .chat-message {
        display: flex;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.3s ease-out;
    }
    
    .chat-message.user {
        justify-content: flex-end;
    }
    
    .chat-bubble {
        max-width: 70%;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        position: relative;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .chat-bubble.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .chat-bubble.assistant {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
        border-bottom-left-radius: 4px;
    }
    
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 0.5rem;
        font-size: 1.2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .chat-avatar.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .chat-avatar.assistant {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        padding: 1rem;
        background: white;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        width: fit-content;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #bbb;
        margin: 0 3px;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    /* Example query cards */
    .query-cards-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
        padding: 0 1rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .query-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .query-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    .query-card:active {
        transform: translateY(0);
    }
    
    .query-card h4 {
        margin: 0 0 0.5rem 0;
        color: #1e3c72;
        font-size: 1.1rem;
    }
    
    .query-card p {
        margin: 0;
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Input area */
    .input-container {
        max-width: 800px;
        margin: 2rem auto;
        padding: 0 1rem;
    }
    
    .stChatInput {
        background: white !important;
        border-radius: 25px !important;
        border: 1px solid #ddd !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInput:focus-within {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
        border-color: #667eea !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0) !important;
    }
    
    .stButton>button:disabled {
        opacity: 0.6 !important;
        cursor: not-allowed !important;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(30, 60, 114, 0.95);
        color: white;
        text-align: center;
        padding: 1rem 0;
        font-size: 0.85rem;
        z-index: 9999;
        backdrop-filter: blur(10px);
    }
    
    /* Animations */
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 1rem !important;
    }
    
    .sidebar-content {
        color: white;
    }
    
    .sidebar-content h3 {
        color: white;
        margin-bottom: 1rem;
    }
    
    .sidebar-content p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background: #4caf50;
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        background: #f44336;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .chat-bubble {
            max-width: 85%;
        }
        
        .app-header h1 {
            font-size: 2rem;
        }
        
        .query-cards-container {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# =============================
# MAIN APPLICATION
# =============================

def main():
    apply_advanced_styling()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🎫 EventTicketing AI</h1>
        <div class="app-subtitle">Your intelligent assistant for all event ticketing needs</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h3>🤖 About</h3>
            <p>This chatbot is specifically designed to handle event ticketing queries using advanced AI models.</p>
            
            <h3>✨ Features</h3>
            <p>• Smart query classification<br>
            • Context-aware responses<br>
            • Entity extraction<br>
            • Real-time processing</p>
            
            <h3>⚡ Status</h3>
            <p><span class="status-indicator status-online"></span>Models Loaded: {status}</p>
            
            <h3>🎯 Scope</h3>
            <p>✅ Ticket bookings<br>
            ✅ Cancellations & refunds<br>
            ✅ Event information<br>
            ✅ Payment issues<br>
            ❌ General conversation</p>
        </div>
        """.format(status="✅" if st.session_state.get('models_loaded', False) else "❌"), 
        unsafe_allow_html=True)

    # Initialize session state
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if "generating" not in st.session_state:
        st.session_state.generating = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "typing" not in st.session_state:
        st.session_state.typing = False

    # Example queries
    example_queries = [
        {"title": "🎫 Buy Tickets", "query": "How do I buy a ticket?"},
        {"title": "🔄 Upgrade Ticket", "query": "How can I upgrade my ticket for the upcoming event in Hyderabad?"},
        {"title": "✏️ Update Details", "query": "How do I change my personal details on my ticket?"},
        {"title": "📅 Find Events", "query": "How can I find details about upcoming events?"},
        {"title": "📞 Contact Support", "query": "How do I contact customer service?"},
        {"title": "💰 Get Refund", "query": "How do I get a refund?"},
        {"title": "❓ Cancellation Fee", "query": "What is the ticket cancellation fee?"},
        {"title": "📊 Track Status", "query": "How can I track my ticket cancellation status?"},
        {"title": "💳 Sell Ticket", "query": "How can I sell my ticket?"}
    ]

    # Model loading
    if not st.session_state.models_loaded:
        with st.spinner("🚀 Initializing AI models... Please wait..."):
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
                    st.toast("✅ Models loaded successfully!", icon="🎉")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Failed to load one or more models. Please refresh the page.")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")

    # Main chat interface
    if st.session_state.models_loaded:
        # Show example query cards
        if not st.session_state.chat_history:
            st.markdown("### 💡 Try these examples:")
            cols = st.columns(2)
            for idx, example in enumerate(example_queries):
                with cols[idx % 2]:
                    if st.button(
                        f"**{example['title']}**\n\n{example['query']}",
                        key=f"example_{idx}",
                        use_container_width=True,
                        disabled=st.session_state.generating
                    ):
                        handle_prompt(example['query'])

        # Chat container
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat history
            for idx, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div class="chat-bubble user">
                            {message['content']}
                        </div>
                        <div class="chat-avatar user">👤</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <div class="chat-avatar assistant">🤖</div>
                        <div class="chat-bubble assistant">
                            {message['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Timestamp
                if 'timestamp' in message:
                    st.markdown(f"""
                    <div style="text-align: {'right' if message['role'] == 'user' else 'left'}; 
                                font-size: 0.75rem; color: #888; 
                                margin: {'0 60px 1rem 0' if message['role'] == 'user' else '0 0 1rem 60px'};">
                        {message['timestamp']}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Typing indicator
            if st.session_state.typing:
                st.markdown("""
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Input area
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        prompt = st.chat_input(
            "Type your ticketing question here...",
            disabled=st.session_state.generating,
            key="chat_input"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Clear chat button
        if st.session_state.chat_history:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button(
                    "🗑️ Clear Chat History",
                    key="clear_chat",
                    disabled=st.session_state.generating,
                    help="Clear all chat messages",
                    use_container_width=True
                ):
                    st.session_state.chat_history = []
                    st.session_state.generating = False
                    st.toast("Chat cleared!", icon="🗑️")
                    time.sleep(0.5)
                    st.rerun()

        # Process user input
        if prompt:
            handle_prompt(prompt)

        # Process generation if needed
        if st.session_state.generating and st.session_state.chat_history:
            process_generation()

# =============================
# INPUT HANDLING FUNCTIONS
# =============================

def handle_prompt(prompt_text):
    if not prompt_text or not prompt_text.strip():
        st.toast("⚠️ Please enter a question!", icon="⚠️")
        return

    st.session_state.generating = True
    st.session_state.typing = True

    # Format and store user message
    prompt_text = prompt_text[0].upper() + prompt_text[1:]
    timestamp = datetime.now().strftime("%H:%M")
    
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt_text,
        "avatar": "👤",
        "timestamp": timestamp
    })
    
    # Rerun to show user message and typing indicator
    st.rerun()

def process_generation():
    if not st.session_state.chat_history:
        return

    last_message = st.session_state.chat_history[-1]["content"]
    
    # Get models from session state
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer

    # Generate response
    if is_ood(last_message, clf_model, clf_tokenizer):
        full_response = random.choice(fallback_responses)
        # Add a small delay to show typing indicator
        time.sleep(0.5)
    else:
        with st.spinner("🤖 Thinking..."):
            dynamic_placeholders = extract_dynamic_placeholders(last_message, nlp)
            response_gpt = generate_response(model, tokenizer, last_message)
            full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
    
    # Hide typing indicator
    st.session_state.typing = False
    
    # Stream the response
    message_placeholder = st.empty()
    streamed_text = ""
    words = full_response.split(" ")
    
    for word in words:
        streamed_text += word + " "
        message_placeholder.markdown(streamed_text + "⬤", unsafe_allow_html=True)
        time.sleep(0.03)  # Faster streaming for better UX
    
    message_placeholder.markdown(full_response, unsafe_allow_html=True)

    # Store assistant response
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": full_response,
        "avatar": "🤖",
        "timestamp": timestamp
    })
    
    # Reset states
    st.session_state.generating = False
    st.session_state.typing = False
    
    # Rerun to update UI
    st.rerun()

# =============================
# FOOTER
# =============================

st.markdown("""
<div class="footer">
    <p>⚠️ This AI is designed specifically for <strong>event ticketing</strong> queries. Responses outside this scope may be inaccurate.</p>
</div>
""", unsafe_allow_html=True)

# =============================
# RUN APP
# =============================

if __name__ == "__main__":
    main()
