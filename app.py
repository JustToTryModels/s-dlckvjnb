import streamlit as st
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSequenceClassification
)
import spacy
import time
import random
from streamlit_lottie import st_lottie
import requests

# =============================
# CONFIGURATION & ASSETS
# =============================

DistilGPT2_MODEL_ID = "IamPradeep/AETCSCB_OOD_IC_DistilGPT2_Fine-tuned"
CLASSIFIER_ID = "IamPradeep/Query_Classifier_DistilBERT"

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie Assets
lottie_bot = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_at6ayqkv.json") # Robot
lottie_loading = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_G8sD8j.json") # Pulse

# Fallback responses (Your original list)
fallback_responses = [
    "I’m sorry, but I am unable to assist with this request. If you need help regarding event tickets, I’d be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    "Unfortunately, I cannot assist with this. However, I am here to help with any event ticket-related concerns you may have."
]

# =============================
# MODEL LOADING (CACHED)
# =============================

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_trf")

@st.cache_resource(show_spinner=False)
def load_gpt2_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained(DistilGPT2_MODEL_ID, trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(DistilGPT2_MODEL_ID)
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_classifier_model():
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID)
    return model, tokenizer

# =============================
# LOGIC FUNCTIONS
# =============================

def is_ood(query: str, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item() == 1

static_placeholders = {
    "{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_TEAM_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    # ... (Keep all your existing placeholders here)
}

def replace_placeholders(response, dynamic, static):
    for k, v in static.items(): response = response.replace(k, v)
    for k, v in dynamic.items(): response = response.replace(k, v)
    return response

def extract_dynamic_placeholders(user_question, nlp):
    doc = nlp(user_question)
    dynamic = {'{{EVENT}}': "event", '{{CITY}}': "city"}
    for ent in doc.ents:
        if ent.label_ == "EVENT": dynamic['{{EVENT}}'] = f"<b>{ent.text.title()}</b>"
        elif ent.label_ == "GPE": dynamic['{{CITY}}'] = f"<b>{ent.text.title()}</b>"
    return dynamic

def generate_response(model, tokenizer, instruction):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, temperature=0.4, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[response.find("Response:") + 9:].strip()

# =============================
# ADVANCED UI / CSS
# =============================

st.set_page_config(page_title="EventBot Pro", page_icon="🎫", layout="centered")

st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [data-testid="stSidebar"], .stApp { font-family: 'Inter', sans-serif !important; }

    /* Glassmorphism Containers */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
        transition: transform 0.3s ease;
    }
    .stChatMessage:hover { transform: translateY(-2px); }

    /* Custom Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 10px 15px -3px rgba(168, 85, 247, 0.4);
        transform: scale(1.02);
    }

    /* Footer Static Warning */
    .footer-note {
        font-size: 0.8rem;
        color: #94a3b8;
        text-align: center;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE & APP FLOW
# =============================

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "generating" not in st.session_state: st.session_state.generating = False

# Sidebar Info & Branding
with st.sidebar:
    st_lottie(lottie_bot, height=150)
    st.title("EventBot AI")
    st.info("I specialize in event ticketing. Ask me about bookings, refunds, or cancellations.")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

# Main Header
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown("# Advanced Event AI 🎫")
    st.markdown("---")

# Model Loading logic
if "models_loaded" not in st.session_state:
    with st.status("Initializing Neural Engines...", expanded=True) as status:
        st.write("Loading Spacy NLP...")
        st.session_state.nlp = load_spacy_model()
        st.write("Loading GPT-2 Head Model...")
        st.session_state.model, st.session_state.tokenizer = load_gpt2_model_and_tokenizer()
        st.write("Loading OOD Classifier...")
        st.session_state.clf_model, st.session_state.clf_tokenizer = load_classifier_model()
        st.session_state.models_loaded = True
        status.update(label="System Ready!", state="complete", expanded=False)
        st.rerun()

# Display Chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Processing Logic
def trigger_ai(text):
    st.session_state.chat_history.append({"role": "user", "content": text})
    st.session_state.generating = True
    st.rerun()

# Input area
if not st.session_state.generating:
    # Quick Actions
    example = st.selectbox("Quick Inquiries:", ["Select a template...", "How do I buy a ticket?", "What is the refund policy?", "I need to cancel my booking"])
    if example != "Select a template...":
        trigger_ai(example)
    
    if prompt := st.chat_input("Ask about your event ticket..."):
        trigger_ai(prompt)
else:
    # Generation Animation
    with st.chat_message("assistant"):
        st_lottie(lottie_loading, height=50, key="loading_anim")
        
        last_user_msg = st.session_state.chat_history[-1]["content"]
        
        # Logic Processing
        if is_ood(last_user_msg, st.session_state.clf_model, st.session_state.clf_tokenizer):
            response = random.choice(fallback_responses)
        else:
            dyn = extract_dynamic_placeholders(last_user_msg, st.session_state.nlp)
            raw = generate_response(st.session_state.model, st.session_state.tokenizer, last_user_msg)
            response = replace_placeholders(raw, dyn, static_placeholders)
        
        # Typing Effect
        placeholder = st.empty()
        full_text = ""
        for chunk in response.split():
            full_text += chunk + " "
            placeholder.markdown(full_text + "▌")
            time.sleep(0.04)
        placeholder.markdown(full_text)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_text})
        st.session_state.generating = False
        st.rerun()

st.markdown('<div class="footer-note">Bot focused strictly on Event Ticketing. Information may vary by organizer.</div>', unsafe_allow_html=True)
