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
    "I apologize, but I cannot help with this request. However, I’d be happy to assist with anything related to event tickets.",
]

# =============================
# MODEL LOADING FUNCTIONS
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
# HELPER FUNCTIONS
# =============================

static_placeholders = { ... }  # your full dictionary (unchanged)

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
            dynamic_placeholders['{{EVENT}}'] = f"<b>{ent.text.title()}</b>"
        elif ent.label_ == "GPE":
            dynamic_placeholders['{{CITY}}'] = f"<b>{ent.text.title()}</b>"
    dynamic_placeholders.setdefault('{{EVENT}}', "event")
    dynamic_placeholders.setdefault('{{CITY}}', "city")
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
            temperature=0.4,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# =============================
# ENHANCED CSS FOR EXTREME INTERACTIVITY
# =============================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif !important; }
    
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
    
    h1 { font-size: 48px !important; text-align: center; background: linear-gradient(90deg, #ff8a00, #e52e71); 
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px !important; }
    
    /* Chat bubbles */
    [data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] {
        background: #f8f9fa !important; border-radius: 20px 20px 20px 5px !important; align-self: flex-start;
    }
    [data-testid="stChatMessage"][data-testid="stChatMessage-user"] {
        background: linear-gradient(90deg, #ff8a00, #e52e71) !important; color: white !important;
        border-radius: 20px 20px 5px 20px !important; align-self: flex-end;
    }
    
    /* Quick buttons - pill style */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 14px 20px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: 100% !important;
        height: 65px !important;
        text-align: left !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3) !important;
    }
    
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%; background: rgba(0,0,0,0.8); color: white;
        text-align: center; padding: 12px; font-size: 14px; z-index: 9999; backdrop-filter: blur(10px);
    }
    
    /* Blinking cursor for typing */
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    .cursor { animation: blink 1s infinite; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎟️ Advanced Event Ticketing Bot 🎉</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px; color: #ddd;'>Your smart assistant for all event ticketing needs</p>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <b>Event Ticketing Assistant Only</b> • Not a general chatbot • Powered by xAI & Hugging Face
</div>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE INIT
# =============================

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "generating" not in st.session_state:
    st.session_state.generating = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================
# LOAD MODELS
# =============================

if not st.session_state.models_loaded:
    with st.spinner("🚀 Waking up the AI engines... This takes a moment..."):
        nlp = load_spacy_model()
        gpt2_model, gpt2_tokenizer = load_gpt2_model_and_tokenizer()
        clf_model, clf_tokenizer = load_classifier_model()
        
        if all([nlp, gpt2_model, gpt2_tokenizer, clf_model, clf_tokenizer]):
            st.session_state.update({
                "models_loaded": True, "nlp": nlp, "model": gpt2_model,
                "tokenizer": gpt2_tokenizer, "clf_model": clf_model, "clf_tokenizer": clf_tokenizer
            })
            st.rerun()
        else:
            st.error("Failed to load models. Please refresh.")

# =============================
# MAIN APP (MODELS LOADED)
# =============================

if st.session_state.models_loaded:
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer

    # Welcome message (only once)
    if not st.session_state.chat_history:
        welcome = """
        **Hey there!** 👋  
        Welcome to the **Advanced Event Ticketing Assistant**!  
        I can help you with:
        - 🎟️ Buying & upgrading tickets  
        - ❌ Cancellations & refunds  
        - 📅 Finding events  
        - 📞 Customer support  
        - and much more!  
        
        **Just click a quick question below or type your own!** 🚀
        """
        st.session_state.chat_history.append({"role": "assistant", "content": welcome, "avatar": "🤖"})

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar=msg["avatar"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # =========================
    # QUICK QUESTION BUTTONS (EXTREMELY INTERACTIVE)
    # =========================
    st.markdown("<br><h3 style='text-align: center; color: #fff;'>🔥 Quick Questions (Click to ask!)</h3>", unsafe_allow_html=True)
    
    quick_queries = [
        "🎟️ How do I buy a ticket?",
        "⬆️ How do I upgrade my ticket?",
        "❌ How do I cancel my ticket?",
        "💰 How do I get a refund?",
        "📞 Contact customer support",
        "📅 Find upcoming events",
        "🏙️ Events in Hyderabad",
        "💳 Cancellation fee?",
        "🎫 How to sell my ticket?"
    ]

    cols = st.columns(3)
    for i, query in enumerate(quick_queries):
        with cols[i % 3]:
            if st.button(query, key=f"quick_{i}", disabled=st.session_state.generating):
                prompt_text = query.split(" ", 1)[1] if query[1] == " " else query
                prompt_text = prompt_text[0].upper() + prompt_text[1:]
                st.session_state.chat_history.append({"role": "user", "content": query, "avatar": "👤"})
                st.session_state.generating = True
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # HANDLE USER INPUT
    # =========================
    def handle_prompt(text):
        if not text.strip():
            return
        clean_text = text.strip()[0].upper() + text.strip()[1:]
        st.session_state.chat_history.append({"role": "user", "content": clean_text, "avatar": "👤"})
        st.session_state.generating = True
        st.rerun()

    if prompt := st.chat_input("Type your question here...", disabled=st.session_state.generating):
        handle_prompt(prompt)

    # =========================
    # GENERATE RESPONSE (WITH TYPING & STREAMING)
    # =========================
    if st.session_state.generating:
        last_user_msg = st.session_state.chat_history[-1]["content"]

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()

            # === Realistic Typing Indicator ===
            typing_msg = "Assistant is typing"
            for _ in range(8):
                for dots in ["", ".", "..", "..."]:
                    placeholder.markdown(f"<i style='color:#888;'>{typing_msg}{dots}</i>", unsafe_allow_html=True)
                    time.sleep(0.25)

            # === Generate Response ===
            if is_ood(last_user_msg, clf_model, clf_tokenizer):
                response = random.choice(fallback_responses)
            else:
                dynamic_ph = extract_dynamic_placeholders(last_user_msg, nlp)
                raw_resp = generate_response(model, tokenizer, last_user_msg)
                response = replace_placeholders(raw_resp, dynamic_ph, static_placeholders)

            # === Character-by-character streaming with blinking cursor ===
            streamed = ""
            for char in response:
                streamed += char
                placeholder.markdown(streamed + "<span class='cursor'>|</span>", unsafe_allow_html=True)
                time.sleep(random.uniform(0.015, 0.07))
            
            placeholder.markdown(streamed, unsafe_allow_html=True)

        # Save final response
        st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})
        st.session_state.generating = False
        st.rerun()

    # Clear chat button
    if st.session_state.chat_history and len(st.session_state.chat_history) > 1:
        if st.button("🗑️ Clear Chat", disabled=st.session_state.generating):
            st.session_state.chat_history = []
            st.rerun()
