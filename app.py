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
from typing import Dict, Tuple, Optional

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Advanced Event Ticketing Chatbot",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
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
    try:
        return spacy.load("en_core_web_trf")
    except Exception:
        # Optional auto-download attempt (works only if environment allows downloads)
        try:
            from spacy.cli import download
            download("en_core_web_trf")
            return spacy.load("en_core_web_trf")
        except Exception as e:
            st.error(
                "spaCy model 'en_core_web_trf' is not available. "
                "Install it with: python -m spacy download en_core_web_trf\n\n"
                f"Error: {e}"
            )
            return None

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

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classify_query(query: str, model, tokenizer) -> Tuple[int, float]:
    """
    Returns:
      pred_id: int
      conf: float (softmax probability of predicted class)
    """
    device = _device()
    model.to(device)
    model.eval()
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_id].item()
    return pred_id, conf

def is_ood(query: str, model, tokenizer) -> bool:
    pred_id, _ = classify_query(query, model, tokenizer)
    return pred_id == 1  # True if OOD (label 1)

# =============================
# ORIGINAL HELPER FUNCTIONS
# =============================

static_placeholders = {
    "{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_TEAM_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CONTACT_SUPPORT_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_CONTACT_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
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
    "{{PAYMENT_METHOD}}": "<b>Payment</b>",
    "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{SUPPORT_ SECTION}}": "<b>Support</b>",  # kept as-is to avoid breaking any existing prompt templates
    "{{CUSTOMER_SUPPORT_SECTION}}": "<b>Customer Support</b>",
    "{{HELP_SECTION}}": "<b>Help</b>",
    "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>",
    "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_BUTTON}}": "<b>Get Refund</b>",
    "{{PAYMENTS_HELP_SECTION}}": "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}": "<b>Payments</b>",
    "{{TICKET_DETAILS}}": "<b>Ticket Details</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>",
    "{{REPORT_PAYMENT_PROBLEM}}": "<b>Report Payment</b>",
    "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{SEND_BUTTON}}": "<b>Send</b>",
    "{{PAYMENT_ISSUE_OPTION}}": "<b>Payment Issue</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}": "<b>Customer Support</b>",
    "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>",
    "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>",
    "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>",
    "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>",
    "{{CURRENT_TICKET_DETAILS}}": "<b>Current Ticket Details</b>",
    "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{CONNECT_WITH_ORGANIZER}}": "<b>Connect with Organizer</b>",
    "{{TICKETS_TAB}}": "<b>Tickets</b>",
    "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>",
}

def replace_placeholders(response: str, dynamic_placeholders: Dict[str, str], static_placeholders_: Dict[str, str]) -> str:
    for placeholder, value in static_placeholders_.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

def extract_dynamic_placeholders(user_question: str, nlp) -> Dict[str, str]:
    dynamic_placeholders: Dict[str, str] = {}

    if nlp is None:
        # Safe fallback
        dynamic_placeholders["{{EVENT}}"] = "event"
        dynamic_placeholders["{{CITY}}"] = "city"
        return dynamic_placeholders

    doc = nlp(user_question)
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders["{{EVENT}}"] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders["{{CITY}}"] = f"<b>{city_text}</b>"

    if "{{EVENT}}" not in dynamic_placeholders:
        dynamic_placeholders["{{EVENT}}"] = "event"
    if "{{CITY}}" not in dynamic_placeholders:
        dynamic_placeholders["{{CITY}}"] = "city"
    return dynamic_placeholders

def generate_response(
    model,
    tokenizer,
    instruction: str,
    max_length: int = 256,
    temperature: float = 0.4,
    top_p: float = 0.95
) -> str:
    model.eval()
    device = _device()
    model.to(device)

    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# =============================
# CSS AND UI SETUP
# =============================

st.markdown(
    """
<style>
/* General typography */
* { font-family: 'Times New Roman', Times, serif !important; }

/* Header */
.hero {
  padding: 18px 18px;
  border-radius: 18px;
  background: linear-gradient(90deg, rgba(255,138,0,0.16), rgba(229,46,113,0.16));
  border: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 12px;
}
.hero h1 { margin: 0; font-size: 43px; line-height: 1.1; }
.hero p { margin: 8px 0 0 0; color: rgba(255,255,255,0.75); }

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg, #ff8a00, #e52e71);
  color: white !important;
  border: none;
  border-radius: 999px;
  padding: 10px 16px;
  font-size: 1.08em;
  font-weight: bold;
  cursor: pointer;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-top: 5px;
  width: 100%;
}
.stButton>button:hover {
  transform: translateY(-1px);
  box-shadow: 0px 10px 25px rgba(0, 0, 0, 0.25);
  color: white !important;
}
.stButton>button:active { transform: translateY(0px) scale(0.99); }

div[data-testid="stChatInput"] {
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.25);
  border-radius: 12px;
  padding: 10px;
  margin: 10px 0;
}

/* Chat bubbles (we wrap content in HTML) */
.bubble {
  padding: 12px 14px;
  border-radius: 14px;
  line-height: 1.5;
  border: 1px solid rgba(255,255,255,0.08);
}
.bubble.user {
  background: linear-gradient(90deg, rgba(41,171,226,0.14), rgba(0,119,182,0.14));
}
.bubble.assistant {
  background: linear-gradient(90deg, rgba(255,138,0,0.12), rgba(229,46,113,0.12));
}
.meta {
  font-size: 12px;
  opacity: 0.7;
  margin-top: 6px;
}

/* Divider between turns */
.horizontal-line { border-top: 2px solid rgba(224,224,224,0.25); margin: 14px 0; }

/* Footer */
.footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background: var(--streamlit-background-color);
  color: gray;
  text-align: center;
  padding: 6px 0;
  font-size: 13px;
  z-index: 9999;
  border-top: 1px solid rgba(255,255,255,0.06);
}
section.main { padding-bottom: 56px; }

/* Subtle "typing" dot */
.typing-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.75);
  margin-left: 6px;
  animation: pulse 0.9s infinite;
}
@keyframes pulse {
  0% { transform: scale(0.8); opacity: 0.35; }
  50% { transform: scale(1.0); opacity: 0.9; }
  100% { transform: scale(0.8); opacity: 0.35; }
}
</style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        This is not a conversational AI. It is designed solely for <b>event ticketing</b> queries. Responses outside this scope may be inaccurate.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="hero">
  <h1>Advanced Event Ticketing Chatbot</h1>
  <p>Ask about ticket bookings, cancellations, refunds, upgrades, payments, ticket transfer/sell, and event-related support.</p>
</div>
""",
    unsafe_allow_html=True
)

# =============================
# STATE INITIALIZATION
# =============================
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "generating" not in st.session_state:
    st.session_state.generating = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "settings" not in st.session_state:
    st.session_state.settings = {
        "max_length": 256,
        "temperature": 0.4,
        "top_p": 0.95,
        "streaming": True,
        "stream_delay": 0.04,   # seconds per word
        "show_timestamps": True,
        "show_debug": False,
    }

def _now_str() -> str:
    return datetime.now().strftime("%H:%M")

def _wrap_bubble(role: str, content: str, ts: Optional[str]) -> str:
    role_class = "user" if role == "user" else "assistant"
    meta = f"<div class='meta'>{ts}</div>" if (ts and st.session_state.settings["show_timestamps"]) else ""
    return f"<div class='bubble {role_class}'>{content}{meta}</div>"

# =============================
# SIDEBAR (INTERACTIVE CONTROLS)
# =============================
with st.sidebar:
    st.markdown("### Controls")
    st.caption("Tune generation, manage chat, and inspect routing (In-Domain vs OOD).")

    # Settings
    with st.expander("Generation settings", expanded=True):
        st.session_state.settings["max_length"] = st.slider(
            "Max response length",
            min_value=96,
            max_value=512,
            value=int(st.session_state.settings["max_length"]),
            step=16,
            disabled=st.session_state.generating,
        )
        st.session_state.settings["temperature"] = st.slider(
            "Creativity (temperature)",
            min_value=0.1,
            max_value=1.2,
            value=float(st.session_state.settings["temperature"]),
            step=0.05,
            disabled=st.session_state.generating,
        )
        st.session_state.settings["top_p"] = st.slider(
            "Top-p",
            min_value=0.5,
            max_value=1.0,
            value=float(st.session_state.settings["top_p"]),
            step=0.01,
            disabled=st.session_state.generating,
        )

    with st.expander("Experience settings", expanded=False):
        st.session_state.settings["streaming"] = st.checkbox(
            "Stream text (typing effect)",
            value=bool(st.session_state.settings["streaming"]),
            disabled=st.session_state.generating,
        )
        st.session_state.settings["stream_delay"] = st.slider(
            "Typing speed (delay per word)",
            min_value=0.00,
            max_value=0.12,
            value=float(st.session_state.settings["stream_delay"]),
            step=0.01,
            disabled=st.session_state.generating or (not st.session_state.settings["streaming"]),
        )
        st.session_state.settings["show_timestamps"] = st.checkbox(
            "Show timestamps",
            value=bool(st.session_state.settings["show_timestamps"]),
            disabled=st.session_state.generating,
        )
        st.session_state.settings["show_debug"] = st.checkbox(
            "Show debug panel (OOD confidence, placeholders)",
            value=bool(st.session_state.settings["show_debug"]),
            disabled=st.session_state.generating,
        )

    st.markdown("---")

    # Conversation tools
    col_a, col_b = st.columns(2)
    with col_a:
        regen = st.button(
            "Regenerate last",
            disabled=st.session_state.generating or len(st.session_state.chat_history) < 2,
            use_container_width=True,
        )
    with col_b:
        clear_side = st.button(
            "Clear chat",
            disabled=st.session_state.generating or len(st.session_state.chat_history) == 0,
            use_container_width=True,
        )

    if clear_side:
        st.session_state.chat_history = []
        st.session_state.generating = False
        st.rerun()

    if regen and st.session_state.chat_history:
        # Remove the last assistant message (if present) and re-generate based on last user message.
        if st.session_state.chat_history[-1].get("role") == "assistant":
            st.session_state.chat_history.pop()
        st.session_state.generating = True
        st.rerun()

    # Edit last user message (interactive)
    with st.expander("Edit last question", expanded=False):
        last_user_idx = None
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            if st.session_state.chat_history[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            st.caption("No user question yet.")
        else:
            current_text = st.session_state.chat_history[last_user_idx].get("content", "")
            edited = st.text_area(
                "Last question",
                value=current_text,
                height=90,
                disabled=st.session_state.generating,
            )
            if st.button(
                "Replace & regenerate",
                disabled=st.session_state.generating or (edited.strip() == ""),
                use_container_width=True,
            ):
                # Drop everything after that user message (so history stays consistent)
                st.session_state.chat_history = st.session_state.chat_history[: last_user_idx + 1]
                st.session_state.chat_history[last_user_idx]["content"] = edited.strip()
                st.session_state.generating = True
                st.rerun()

    st.markdown("---")

    # Export transcript
    with st.expander("Export transcript", expanded=False):
        if st.session_state.chat_history:
            as_md_lines = []
            for m in st.session_state.chat_history:
                role = m.get("role", "unknown")
                ts = m.get("ts", "")
                content = m.get("content", "")
                as_md_lines.append(f"**{role.upper()}** ({ts})\n\n{content}\n")
            md_text = "\n---\n".join(as_md_lines)

            st.download_button(
                "Download (.md)",
                data=md_text.encode("utf-8"),
                file_name="chat_transcript.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=st.session_state.generating,
            )
        else:
            st.caption("Nothing to export yet.")

    st.markdown("---")
    st.caption(f"Runtime device: `{_device()}`")

# =============================
# EXAMPLES
# =============================
example_queries = [
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
# MODEL LOADING
# =============================
if not st.session_state.models_loaded:
    with st.spinner("Loading models and resources..."):
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
                st.error("Failed to load one or more models. Please refresh the page.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

# ==================================
# MAIN CHAT INTERFACE
# ==================================

if st.session_state.models_loaded:
    # Header helper row
    top_left, top_right = st.columns([0.72, 0.28], vertical_alignment="center")
    with top_left:
        st.write("Ask me about ticket bookings, cancellations, refunds, or any event-related inquiries!")
    with top_right:
        # Quick stats (interactive feel)
        turns = sum(1 for m in st.session_state.chat_history if m.get("role") == "user")
        st.metric("Conversation turns", turns)

    # Example chooser (kept) + quick buttons (added)
    selected_query = st.selectbox(
        "Choose a query from examples:",
        ["Choose your question"] + example_queries,
        key="query_selectbox",
        label_visibility="collapsed",
        disabled=st.session_state.generating
    )

    c1, c2, c3 = st.columns([0.34, 0.33, 0.33], vertical_alignment="center")
    with c1:
        process_query_button = st.button(
            "Ask selected example",
            key="query_button",
            disabled=st.session_state.generating,
            use_container_width=True,
        )
    with c2:
        st.button(
            "Regenerate",
            key="regen_main",
            disabled=st.session_state.generating or len(st.session_state.chat_history) < 2,
            use_container_width=True,
            on_click=lambda: (
                st.session_state.chat_history.pop()
                if (st.session_state.chat_history and st.session_state.chat_history[-1].get("role") == "assistant")
                else None,
                setattr(st.session_state, "generating", True)
            ),
        )
    with c3:
        st.button(
            "Clear",
            key="clear_main",
            disabled=st.session_state.generating or len(st.session_state.chat_history) == 0,
            use_container_width=True,
            on_click=lambda: (
                st.session_state.chat_history.clear(),
                setattr(st.session_state, "generating", False)
            ),
        )

    with st.expander("Quick-pick examples (one click)", expanded=False):
        # A grid of example buttons for a more interactive UI
        rows = [example_queries[i:i+3] for i in range(0, len(example_queries), 3)]
        for r_i, row in enumerate(rows):
            cols = st.columns(3)
            for j, q in enumerate(row):
                if cols[j].button(q, key=f"exbtn_{r_i}_{j}", disabled=st.session_state.generating, use_container_width=True):
                    # Same flow as any prompt
                    if q.strip():
                        # call handle_prompt below (defined later) by setting a session var trigger
                        st.session_state["_queued_prompt"] = q
                        st.rerun()

    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer

    last_role = None

    # Display chat history
    for idx, message in enumerate(st.session_state.chat_history):
        role = message.get("role", "assistant")
        avatar = message.get("avatar", "🤖" if role == "assistant" else "👤")
        content = message.get("content", "")
        ts = message.get("ts", "")

        if role == "user" and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

        with st.chat_message(role, avatar=avatar):
            st.markdown(_wrap_bubble(role, content, ts), unsafe_allow_html=True)

            # Per-message feedback (interactive)
            if role == "assistant":
                fb_cols = st.columns([0.18, 0.18, 0.64])
                with fb_cols[0]:
                    if st.button("👍", key=f"up_{idx}", disabled=st.session_state.generating):
                        message["feedback"] = "up"
                        st.toast("Thanks! Feedback saved.")
                with fb_cols[1]:
                    if st.button("👎", key=f"down_{idx}", disabled=st.session_state.generating):
                        message["feedback"] = "down"
                        st.toast("Got it. Feedback saved.")
                with fb_cols[2]:
                    if message.get("feedback"):
                        st.caption(f"Feedback: {message['feedback']}")

        last_role = role

    # --- Unified function to handle prompt processing ---
    def handle_prompt(prompt_text: str):
        if not prompt_text or not prompt_text.strip():
            st.toast("⚠️ Please enter or select a question.")
            return

        # Lock the UI
        st.session_state.generating = True

        prompt_text = prompt_text.strip()
        prompt_text = prompt_text[0].upper() + prompt_text[1:] if prompt_text else prompt_text

        st.session_state.chat_history.append(
            {"role": "user", "content": prompt_text, "avatar": "👤", "ts": _now_str()}
        )

        # Rerun to display user's message and disable inputs immediately
        st.rerun()

    def process_generation():
        # Runs only after UI is locked and the user message is shown
        last_message = st.session_state.chat_history[-1]["content"]

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()

            # Small interactive "status" feel
            if hasattr(st, "status"):
                status = st.status("Working…", expanded=False)
            else:
                status = None

            t0 = time.time()

            pred_id, conf = classify_query(last_message, clf_model, clf_tokenizer)
            ood = (pred_id == 1)

            if status:
                status.update(label="Routing query (In-domain vs OOD)…", state="running")

            debug_panel = ""
            if st.session_state.settings["show_debug"]:
                debug_panel += (
                    f"<div class='meta'>"
                    f"Classifier: pred={pred_id} ({'OOD' if ood else 'In-domain'}), conf={conf:.3f} • "
                    f"Device={_device()}</div>"
                )

            if ood:
                full_response = random.choice(fallback_responses)
            else:
                if status:
                    status.update(label="Generating answer…", state="running")

                dynamic_placeholders = extract_dynamic_placeholders(last_message, nlp)
                response_gpt = generate_response(
                    model,
                    tokenizer,
                    last_message,
                    max_length=int(st.session_state.settings["max_length"]),
                    temperature=float(st.session_state.settings["temperature"]),
                    top_p=float(st.session_state.settings["top_p"]),
                )
                full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)

                if st.session_state.settings["show_debug"]:
                    debug_panel += (
                        "<div class='meta'>Dynamic placeholders: "
                        + ", ".join([f"{k}={v}" for k, v in dynamic_placeholders.items()])
                        + "</div>"
                    )

            # Stream output for interactivity
            if st.session_state.settings["streaming"] and st.session_state.settings["stream_delay"] > 0:
                streamed_text = ""
                words = full_response.split(" ")
                prog = st.progress(0, text="Typing…")
                total = max(len(words), 1)

                for i, word in enumerate(words, start=1):
                    streamed_text += word + " "
                    bubble = _wrap_bubble(
                        "assistant",
                        streamed_text.strip() + " <span class='typing-dot'></span>" + debug_panel,
                        _now_str() if st.session_state.settings["show_timestamps"] else None
                    )
                    message_placeholder.markdown(bubble, unsafe_allow_html=True)
                    prog.progress(int((i / total) * 100), text="Typing…")
                    time.sleep(float(st.session_state.settings["stream_delay"]))

                prog.empty()
                message_placeholder.markdown(
                    _wrap_bubble("assistant", full_response + debug_panel, _now_str()),
                    unsafe_allow_html=True
                )
            else:
                # Instant render (still interactive via debug + layout)
                message_placeholder.markdown(
                    _wrap_bubble("assistant", full_response + debug_panel, _now_str()),
                    unsafe_allow_html=True
                )

            if status:
                dt = time.time() - t0
                status.update(label=f"Done in {dt:.2f}s", state="complete")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response, "avatar": "🤖", "ts": _now_str()}
        )

        # Unlock UI
        st.session_state.generating = False

    # --- LOGIC FLOW ---
    # 0. Handle queued prompt from quick buttons
    if "_queued_prompt" in st.session_state and st.session_state["_queued_prompt"]:
        qp = st.session_state["_queued_prompt"]
        st.session_state["_queued_prompt"] = ""
        handle_prompt(qp)

    # 1. Handle triggers (button click or chat input)
    if process_query_button:
        if selected_query != "Choose your question":
            handle_prompt(selected_query)
        else:
            st.error("⚠️ Please select your question from the dropdown.")

    if prompt := st.chat_input("Enter your own question:", disabled=st.session_state.generating):
        handle_prompt(prompt)

    # 2. If UI is locked, generate a response
    if st.session_state.generating:
        process_generation()
        st.rerun()
