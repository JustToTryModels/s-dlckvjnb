import streamlit as st
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSequenceClassification
)
import spacy
import time
import random

# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(
    page_title="EventGenie AI",
    page_icon="🎫",
    layout="centered",
    initial_sidebar_state="expanded"
)

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
# ORIGINAL HELPER FUNCTIONS
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
# CSS AND UI SETUP (EXTREMELY INTERACTIVE)
# =============================

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        /* General Font and Body */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif !important;
        }
        
        /* Main Container Background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        /* Custom Title Styling */
        .title-container {
            text-align: center;
            padding: 20px;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin-bottom: 30px;
        }
        .title-text {
            font-size: 3em;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #0077B6, #e52e71);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle-text {
            color: #555;
            font-size: 1.1em;
            margin-top: -10px;
        }

        /* Interactive Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white !important;
            border: none;
            border-radius: 50px;
            padding: 12px 28px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease-in-out;
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            background: linear-gradient(90deg, #182848 0%, #4b6cb7 100%);
        }
        
        /* Secondary Button (Clear Chat) */
        div[data-testid="stSidebar"] .stButton>button {
            background: linear-gradient(90deg, #ff416c, #ff4b2b);
        }

        /* Chat Input Styling */
        div[data-testid="stChatInput"] {
            background: white;
            border-radius: 20px;
            box-shadow: 0 -5px 20px rgba(0,0,0,0.1);
            border: 2px solid #e0e0e0;
        }

        /* Message Bubbles Animation */
        .element-container {
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Horizontal Line */
        .horizontal-line {
            height: 2px;
            background: linear-gradient(to right, transparent, #ccc, transparent);
            margin: 25px 0;
        }

        /* Footer */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: rgba(255, 255, 255, 0.9);
            color: #666;
            text-align: center;
            padding: 10px 0;
            font-size: 12px;
            z-index: 9999;
            border-top: 1px solid #eaeaea;
            backdrop-filter: blur(5px);
        }
        .main { padding-bottom: 60px; }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            font-weight: 600;
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        🔒 Designed solely for <b>event ticketing</b> queries. Responses outside this scope may be inaccurate.
    </div>
    """,
    unsafe_allow_html=True
)

# =============================
# HEADER SECTION
# =============================
st.markdown(
    """
    <div class="title-container">
        <div class="title-text">EventGenie AI</div>
        <div class="subtitle-text">Your Intelligent Ticketing Assistant</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- INITIALIZE STATE VARIABLES ---
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "generating" not in st.session_state:
    st.session_state.generating = False 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Add a welcome message to history if empty
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "👋 **Hello!** I'm here to help with your tickets, events, and refunds. How can I assist you today?",
        "avatar": "🤖"
    })

example_queries = [
    "How do I buy a ticket?", "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?", "How can I find details about upcoming events?",
    "How do I contact customer service?", "How do I get a refund?", "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation status?", "How can I sell my ticket?"
]

# =============================
# LOADING SCREEN
# =============================
if not st.session_state.models_loaded:
    with st.container():
        st.info("🚀 Initializing AI Core...")
        progress_bar = st.progress(0)
        
        try:
            nlp = load_spacy_model()
            progress_bar.progress(30)
            
            gpt2_model, gpt2_tokenizer = load_gpt2_model_and_tokenizer()
            progress_bar.progress(70)
            
            clf_model, clf_tokenizer = load_classifier_model()
            progress_bar.progress(100)

            if all([nlp, gpt2_model, gpt2_tokenizer, clf_model, clf_tokenizer]):
                st.session_state.models_loaded = True
                st.session_state.nlp = nlp
                st.session_state.model = gpt2_model
                st.session_state.tokenizer = gpt2_tokenizer
                st.session_state.clf_model = clf_model
                st.session_state.clf_tokenizer = clf_tokenizer
                time.sleep(0.5) # Aesthetic pause
                st.rerun()
            else:
                st.error("Failed to load one or more models. Please refresh the page.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

# ==================================
# MAIN CHAT INTERFACE
# ==================================

if st.session_state.models_loaded:
    
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        st.markdown("Use this panel to manage your session.")
        
        # Clear chat button in sidebar
        if st.button("🗑️ Clear Conversation", key="reset_button", disabled=st.session_state.generating):
            st.session_state.chat_history = []
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "👋 **Hello!** I'm here to help with your tickets, events, and refunds. How can I assist you today?",
                "avatar": "🤖"
            })
            st.session_state.generating = False
            st.toast("Conversation cleared!", icon="🧹")
            st.rerun()
            
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This AI uses a fine-tuned **DistilGPT2** model for generation and **DistilBERT** for query classification."
        )

    # --- QUICK QUESTIONS SECTION ---
    with st.expander("📝 **Quick Questions (Click to Expand)**", expanded=False):
        st.markdown("Don't know what to ask? Try one of these:")
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_query = st.selectbox(
                "Select a query:", ["Choose a question..."] + example_queries,
                key="query_selectbox", label_visibility="collapsed",
                disabled=st.session_state.generating
            )
        with col2:
            process_query_button = st.button(
                "Ask ➡️", key="query_button",
                disabled=st.session_state.generating
            )

    # --- CHAT HISTORY DISPLAY ---
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer

    last_role = None

    for message in st.session_state.chat_history:
        # Visual separator between turns, except for the very first welcome message
        if message["role"] == "user" and last_role == "assistant":
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
             
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"]

    # --- FUNCTION: Handle Prompt ---
    def handle_prompt(prompt_text):
        if not prompt_text or not prompt_text.strip() or prompt_text == "Choose a question...":
            st.toast("⚠️ Please enter or select a valid question.", icon="🚫")
            return

        st.session_state.generating = True
        
        # Format input
        prompt_text = prompt_text[0].upper() + prompt_text[1:]
        st.session_state.chat_history.append({"role": "user", "content": prompt_text, "avatar": "👤"})
        
        st.rerun()

    # --- FUNCTION: Process Generation ---
    def process_generation():
        last_message = st.session_state.chat_history[-1]["content"]

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            # Visual "Thinking" indicator
            with st.status("Thinking...", expanded=True) as status:
                st.write("Analyzing your request...")
                time.sleep(0.5) # UX pause
                
                if is_ood(last_message, clf_model, clf_tokenizer):
                    status.update(label="Topic outside scope", state="error", expanded=False)
                    full_response = random.choice(fallback_responses)
                else:
                    status.update(label="Drafting response", state="running", expanded=False)
                    dynamic_placeholders = extract_dynamic_placeholders(last_message, nlp)
                    response_gpt = generate_response(model, tokenizer, last_message)
                    full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                    status.update(label="Complete!", state="complete", expanded=False)

            # Typewriter effect
            streamed_text = ""
            for word in full_response.split(" "):
                streamed_text += word + " "
                message_placeholder.markdown(streamed_text + "⬤", unsafe_allow_html=True)
                time.sleep(0.04) # Slightly faster for better feel
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        st.session_state.generating = False

    # --- LOGIC FLOW ---
    
    # 1. Handle Example Trigger
    if process_query_button:
        handle_prompt(selected_query)

    # 2. Handle Chat Input
    if prompt := st.chat_input("Type your question here...", disabled=st.session_state.generating):
        handle_prompt(prompt)

    # 3. Generate if locked
    if st.session_state.generating:
        process_generation()
        st.rerun()
