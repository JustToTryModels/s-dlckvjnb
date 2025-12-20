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
# MODEL AND CONFIGURATION SETUP
# =============================

DistilGPT2_MODEL_ID = "IamPradeep/AETCSCB_OOD_IC_DistilGPT2_Fine-tuned"
CLASSIFIER_ID = "IamPradeep/Query_Classifier_DistilBERT"

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
    "I apologize, but I cannot help with this request. However, I’d be happy to assist with anything related to event tickets."
]

# =============================
# GLOBAL CSS WITH CONSISTENT FONT SIZE
# =============================

st.markdown("""
<style>
    /* Global Font & Size */
    html, body, [class*="css"] {
        font-family: 'Tiempos Text', Georgia, serif !important;
        font-size: 18px !important;
        line-height: 1.6;
    }

    /* Title */
    .main-title {
        font-family: 'Tiempos', 'Tiempos Headline', Georgia, serif !important;
        font-size: 48px !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
        color: #1a1a1a;
    }

    /* Chat messages */
    .stChatMessage {
        font-size: 18px !important;
    }

    /* Input & Selectbox */
    div[data-testid="stTextInput"] input,
    div[data-testid="stChatInput"] textarea,
    .stSelectbox > div > div {
        font-size: 18px !important;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Tiempos Text', Georgia, serif !important;
        font-size: 18px !important;
        font-weight: 600;
        background: linear-gradient(90deg, #ff8a00, #e52e71) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 12px 28px !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(229, 46, 113, 0.4) !important;
    }

    /* Example query button variant */
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
        background: linear-gradient(90deg, #29ABE2, #0077B6) !important;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f9f9f9;
        color: #666;
        text-align: center;
        padding: 12px 0;
        font-size: 15px !important;
        z-index: 9999;
        border-top: 1px solid #eee;
    }

    /* Horizontal line between messages */
    .horizontal-line {
        border-top: 1px solid #e0e0e0;
        margin: 20px 0;
    }

    /* Spinner and toast */
    .stSpinner > div {
        font-size: 18px !important;
    }

    /* Reduce padding on main container for better spacing */
    .main > div {
        padding-top: 2rem;
        padding-bottom: 6rem;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    This chatbot is designed <strong>exclusively for event ticketing</strong> support. 
    Queries outside this scope will receive limited responses.
</div>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">Advanced Event Ticketing Chatbot</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 20px;'>Your smart assistant for bookings, cancellations, refunds, and more.</p>", unsafe_allow_html=True)

# =============================
# MODEL LOADING
# =============================

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_trf")

@st.cache_resource(show_spinner=False)
def load_gpt2_model_and_tokenizer():
    try:
        model = GPT2LMHeadModel.from_pretrained(DistilGPT2_MODEL_ID, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(DistilGPT2_MODEL_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load GPT-2 model: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_classifier_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")
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
# PLACEHOLDERS & HELPERS
# =============================

static_placeholders = {
    "{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_TEAM_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CONTACT_SUPPORT_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CANCEL_TICKET_SECTION}}": "**Ticket Cancellation**",
    "{{GET_REFUND_OPTION}}": "**Get Refund**",
    "{{UPGRADE_TICKET_INFORMATION}}": "**Upgrade Ticket Information**",
    "{{TICKET_SECTION}}": "**Ticketing**",
    "{{CANCELLATION_POLICY_SECTION}}": "**Cancellation Policy**",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "**Check Cancellation Policy**",
    "{{CHECK_REFUND_POLICY_OPTION}}": "**Check Refund Policy**",
    "{{CHECK_PRIVACY_POLICY_OPTION}}": "**Check Privacy Policy**",
    "{{SAVE_BUTTON}}": "**Save**",
    "{{EDIT_BUTTON}}": "**Edit**",
    "{{CANCELLATION_FEE_SECTION}}": "**Cancellation Fee**",
    "{{PRIVACY_POLICY_LINK}}": "**Privacy Policy**",
    "{{REFUND_SECTION}}": "**Refund**",
    "{{REFUND_POLICY_LINK}}": "**Refund Policy**",
    "{{CUSTOMER_SERVICE_SECTION}}": "**Customer Service**",
    "{{EVENT_ORGANIZER_OPTION}}": "**Event Organizer**",
    "{{FIND_TICKET_OPTION}}": "**Find Ticket**",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "**Find Upcoming Events**",
    "{{CONTACT_SECTION}}": "**Contact**",
    "{{SEARCH_BUTTON}}": "**Search**",
    "{{SUPPORT_SECTION}}": "**Support**",
    "{{EVENTS_SECTION}}": "**Events**",
    "{{PAYMENT_SECTION}}": "**Payment**",
    "{{TRANSFER_TICKET_OPTION}}": "**Transfer Ticket**",
    "{{SELL_TICKET_OPTION}}": "**Sell Ticket**",
    "{{UPGRADE_TICKET_OPTION}}": "**Upgrade Ticket**",
    "{{CANCEL_TICKET_BUTTON}}": "**Cancel Ticket**",
    "{{GET_REFUND_BUTTON}}": "**Get Refund**",
    "{{TRANSFER_TICKET_BUTTON}}": "**Transfer Ticket**",
    "{{CONNECT_WITH_ORGANIZER}}": "**Connect with Organizer**",
    # Add more as needed...
}

def replace_placeholders(response, dynamic_placeholders):
    for placeholder, value in {**static_placeholders, **dynamic_placeholders}.items():
        response = response.replace(placeholder, value)
    return response

def extract_dynamic_placeholders(user_question, nlp):
    doc = nlp(user_question)
    dynamic = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            dynamic['{{EVENT}}'] = f"**{ent.text.title()}**"
        elif ent.label_ == "GPE":
            dynamic['{{CITY}}'] = f"**{ent.text.title()}**"
    dynamic.setdefault('{{EVENT}}', 'event')
    dynamic.setdefault('{{CITY}}', 'city')
    return dynamic

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
            temperature=0.4,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# =============================
# SESSION STATE INIT
# =============================

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "generating" not in st.session_state:
    st.session_state.generating = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the concert in Hyderabad?",
    "How do I cancel my ticket and get a refund?",
    "What is the cancellation fee?",
    "How can I contact customer support?",
    "How do I sell my extra ticket?",
    "When will I receive my e-ticket?"
]

# =============================
# LOAD MODELS
# =============================

if not st.session_state.models_loaded:
    with st.spinner("Loading advanced AI models... This may take a moment."):
        nlp = load_spacy_model()
        gpt2_model, gpt2_tokenizer = load_gpt2_model_and_tokenizer()
        clf_model, clf_tokenizer = load_classifier_model()

        if all([nlp, gpt2_model, gpt2_tokenizer, clf_model, clf_tokenizer]):
            st.session_state.update({
                "models_loaded": True,
                "nlp": nlp,
                "model": gpt2_model,
                "tokenizer": gpt2_tokenizer,
                "clf_model": clf_model,
                "clf_tokenizer": clf_tokenizer
            })
            st.success("Models loaded successfully!")
            st.rerun()
        else:
            st.error("Failed to load required models. Please refresh the page.")

# =============================
# MAIN CHAT UI
# =============================

if st.session_state.models_loaded:
    st.markdown("### Ask me anything about event tickets, bookings, cancellations, or refunds!")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_query = st.selectbox(
            "Quick examples:", ["Type your own question..."] + example_queries,
            label_visibility="collapsed",
            disabled=st.session_state.generating
        )
    with col2:
        ask_button = st.button(
            "Ask Question ➤" if selected_query != "Type your own question..." else "Send",
            use_container_width=True,
            disabled=st.session_state.generating or (selected_query == "Type your own question...")
        )

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i > 0 and message["role"] == "assistant" and st.session_state.chat_history[i-1]["role"] == "user":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Input handlers
    if ask_button and selected_query != "Type your own question...":
        user_input = selected_query
    elif prompt := st.chat_input("Type your question here...", disabled=st.session_state.generating):
        user_input = prompt
    else:
        user_input = None

    if user_input:
        st.session_state.generating = True
        user_input = user_input.strip()
        if user_input:
            user_input = user_input[0].upper() + user_input[1:]
            st.session_state.chat_history.append({"role": "user", "content": user_input, "avatar": "User"})
            st.rerun()

    # Generate response when needed
    if st.session_state.generating and st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        query = st.session_state.chat_history[-1]["content"]

        with st.chat_message("assistant", avatar="Robot"):
            placeholder = st.empty()
            full_response = ""

            if is_ood(query, st.session_state.clf_model, st.session_state.clf_tokenizer):
                full_response = random.choice(fallback_responses)
            else:
                with st.spinner("Thinking..."):
                    dynamic_ph = extract_dynamic_placeholders(query, st.session_state.nlp)
                    raw_resp = generate_response(st.session_state.model, st.session_state.tokenizer, query)
                    full_response = replace_placeholders(raw_resp, dynamic_ph)

            # Stream response
            streamed = ""
            for word in full_response.split():
                streamed += word + " "
                placeholder.markdown(streamed + "▌")
                time.sleep(0.03)
            placeholder.markdown(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "Robot"})
        st.session_state.generating = False
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Conversation", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.session_state.generating = False
            st.rerun()
