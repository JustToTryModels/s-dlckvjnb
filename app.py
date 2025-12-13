import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONFIGURATION
# =============================================================================
HF_REPO_ID = "Zlib2/RFC"
MODEL_FILENAME = "final_random_forest_model.joblib"

BEST_FEATURES = [
    "satisfaction_level",
    "time_spend_company",
    "average_monthly_hours",
    "number_project",
    "last_evaluation",
]

# =============================================================================
# STYLING
# =============================================================================
def inject_css() -> None:
    st.markdown(
        """
<style>
/* ---- Typography ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg: #F6F8FC;
  --panel: rgba(255,255,255,0.78);
  --panel-solid: #FFFFFF;
  --stroke: rgba(15, 23, 42, 0.10);

  --text: #0F172A;       /* slate-900 */
  --muted: #475569;      /* slate-600 */

  --brand: #1E3A8A;      /* blue-800 */
  --brand2: #0EA5E9;     /* sky-500 */
  --accent: #14B8A6;     /* teal-500 */

  --good: #16A34A;       /* green-600 */
  --bad: #DC2626;        /* red-600 */
  --warn: #F59E0B;       /* amber-500 */

  --shadow: 0 18px 45px rgba(15,23,42,.12);
  --shadow-soft: 0 10px 25px rgba(15,23,42,.10);
  --radius: 18px;
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
}

/* ---- App background ---- */
.stApp {
  background:
    radial-gradient(1200px 600px at 10% 10%, rgba(14,165,233,.12), transparent 60%),
    radial-gradient(900px 500px at 90% 20%, rgba(20,184,166,.10), transparent 55%),
    linear-gradient(180deg, var(--bg), var(--bg));
  color: var(--text);
}

/* ---- Hide Streamlit chrome (optional) ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(30,58,138,.10), rgba(255,255,255,0.0));
  border-right: 1px solid var(--stroke);
}
section[data-testid="stSidebar"] > div {
  padding-top: 1.25rem;
}
.sidebar-card {
  background: rgba(255,255,255,0.70);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 10px 25px rgba(15,23,42,.06);
}

/* ---- Layout containers ---- */
.hero {
  border-radius: var(--radius);
  padding: 26px 26px;
  background: linear-gradient(135deg, rgba(30,58,138,.10), rgba(14,165,233,.10), rgba(20,184,166,.08));
  border: 1px solid var(--stroke);
  box-shadow: var(--shadow-soft);
  position: relative;
  overflow: hidden;
  animation: fadeUp .55s ease both;
}
.hero:before{
  content:"";
  position:absolute;
  inset:-2px;
  background:
    radial-gradient(600px 160px at 20% 20%, rgba(255,255,255,.55), transparent 60%),
    radial-gradient(500px 140px at 80% 10%, rgba(255,255,255,.35), transparent 60%);
  pointer-events:none;
  filter: blur(0.2px);
}
.hero-inner{
  position:relative;
  z-index:1;
}
.brand-row{
  display:flex;
  gap:14px;
  align-items:center;
}
.logo-badge{
  width:44px; height:44px;
  border-radius: 14px;
  background: linear-gradient(135deg, var(--brand), var(--brand2));
  box-shadow: 0 10px 25px rgba(30,58,138,.20);
  display:flex;
  align-items:center;
  justify-content:center;
  color: white;
  font-weight: 800;
  letter-spacing: 0.5px;
  user-select:none;
}
.hero-title{
  font-size: 2.1rem;
  line-height: 1.2;
  font-weight: 800;
  margin: 0;
  color: var(--text);
}
.hero-sub{
  margin-top: 6px;
  color: var(--muted);
  font-size: 1.03rem;
  max-width: 70ch;
}
.badges{
  margin-top: 14px;
  display:flex;
  gap:10px;
  flex-wrap: wrap;
}
.pill{
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,.62);
  padding: 7px 10px;
  border-radius: 999px;
  color: var(--muted);
  font-size: .92rem;
}

/* ---- Cards / sections ---- */
.card{
  background: var(--panel);
  backdrop-filter: blur(10px);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  padding: 18px 18px;
  animation: fadeUp .45s ease both;
}
.card h3{
  margin: 0 0 8px 0;
  font-size: 1.05rem;
  color: var(--text);
}
.card .hint{
  color: var(--muted);
  font-size: .95rem;
  margin-bottom: 10px;
}

/* ---- Inputs polish ---- */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea{
  border-radius: 14px !important;
  border: 1px solid rgba(15,23,42,.14) !important;
  padding: 10px 12px !important;
  background: rgba(255,255,255,.75) !important;
}

div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus{
  border: 1px solid rgba(14,165,233,.60) !important;
  box-shadow: 0 0 0 4px rgba(14,165,233,.15) !important;
}

/* Baseweb slider styling */
div[data-testid="stSlider"] [data-baseweb="slider"] > div{
  padding-top: 6px;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBar"]{
  opacity: .25;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  box-shadow: 0 10px 25px rgba(15,23,42,.18) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background: linear-gradient(135deg, var(--brand2), var(--accent)) !important;
  border: none !important;
}

/* ---- CTA button: professional gradient + shimmer ---- */
.stButton>button {
  width: 100%;
  border: 0 !important;
  border-radius: 16px !important;
  padding: 14px 18px !important;
  font-weight: 800 !important;
  letter-spacing: .2px;
  color: #fff !important;
  background: linear-gradient(135deg, var(--brand), var(--brand2)) !important;
  box-shadow: 0 14px 35px rgba(30,58,138,.24);
  transform: translateY(0);
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
  position: relative;
  overflow: hidden;
}

.stButton>button::before{
  content:"";
  position:absolute;
  top:0; left:-120%;
  width: 60%;
  height:100%;
  background: linear-gradient(120deg, transparent, rgba(255,255,255,.35), transparent);
  transform: skewX(-18deg);
  transition: left .6s ease;
}

.stButton>button:hover{
  transform: translateY(-2px);
  box-shadow: 0 18px 45px rgba(30,58,138,.28);
  filter: saturate(1.05);
}
.stButton>button:hover::before{
  left: 140%;
}
.stButton>button:active{
  transform: translateY(1px) scale(.99);
  box-shadow: 0 10px 22px rgba(30,58,138,.22);
}

/* ---- Result cards ---- */
.result-good{
  border: 1px solid rgba(22,163,74,.25);
  background: linear-gradient(180deg, rgba(22,163,74,.10), rgba(255,255,255,.65));
}
.result-bad{
  border: 1px solid rgba(220,38,38,.25);
  background: linear-gradient(180deg, rgba(220,38,38,.10), rgba(255,255,255,.65));
}
.result-title{
  font-size: 1.45rem;
  font-weight: 900;
  margin: 0 0 8px 0;
}
.result-sub{
  margin: 0;
  color: var(--muted);
}

/* ---- Probability bars ---- */
.bar-wrap{
  width: 100%;
  height: 14px;
  background: rgba(15,23,42,.08);
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid rgba(15,23,42,.10);
}
.bar{
  height: 100%;
  width: 0%;
  border-radius: 999px;
  transition: width .7s cubic-bezier(.2,.8,.2,1);
}
.bar-good{ background: linear-gradient(90deg, rgba(22,163,74,.90), rgba(34,197,94,.85)); }
.bar-bad{  background: linear-gradient(90deg, rgba(220,38,38,.90), rgba(239,68,68,.85)); }

/* ---- Expander polish ---- */
div[data-testid="stExpander"] details summary{
  background: rgba(255,255,255,.70) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: 14px !important;
  padding: .85rem 1rem !important;
}
div[data-testid="stExpander"] details[open] summary{
  border-bottom-left-radius: 0 !important;
  border-bottom-right-radius: 0 !important;
}
div[data-testid="stExpander"] details > div{
  border: 1px solid var(--stroke) !important;
  border-top: none !important;
  border-bottom-left-radius: 14px !important;
  border-bottom-right-radius: 14px !important;
  background: rgba(255,255,255,.65) !important;
}

/* ---- Animations ---- */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ---- Responsive tweaks ---- */
@media (max-width: 900px){
  .hero-title{ font-size: 1.7rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_model_from_huggingface():
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# =============================================================================
# UI HELPERS
# =============================================================================
def sync_state(dst_key: str, src_key: str) -> None:
    st.session_state[dst_key] = st.session_state[src_key]


def paired_float_control(
    *,
    title: str,
    state_key: str,
    slider_key: str,
    input_key: str,
    min_value: float,
    max_value: float,
    step: float,
    help_text: str,
) -> float:
    st.write(f"**{title}**")
    c1, c2 = st.columns([3, 1], vertical_alignment="center")

    with c1:
        st.slider(
            title,
            min_value=min_value,
            max_value=max_value,
            value=float(st.session_state[state_key]),
            step=step,
            help=help_text,
            label_visibility="collapsed",
            key=slider_key,
            on_change=lambda: sync_state(state_key, slider_key),
        )
    with c2:
        st.number_input(
            title,
            min_value=min_value,
            max_value=max_value,
            value=float(st.session_state[state_key]),
            step=step,
            format="%.2f",
            help=help_text,
            label_visibility="collapsed",
            key=input_key,
            on_change=lambda: sync_state(state_key, input_key),
        )

    return float(st.session_state[state_key])


def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="hero-inner">
    <div class="brand-row">
      <div class="logo-badge">HR</div>
      <div>
        <div class="hero-title">Employee Turnover Prediction</div>
        <div class="hero-sub">Enter a few key signals to estimate turnover risk, then review probabilities and a transparent input summary.</div>
      </div>
    </div>
    <div class="badges">
      <div class="pill">Random Forest • Balanced</div>
      <div class="pill">5 selected features</div>
      <div class="pill">Fast, interactive prediction</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### Model & App")
        st.caption("A lightweight, professional UI for HR turnover risk screening.")

        st.markdown(
            f"""
**Repository:** `{HF_REPO_ID}`  
**Model:** Random Forest Classifier  
**Features:** {len(BEST_FEATURES)}  
**Target:** 0 = Stay, 1 = Leave
            """.strip()
        )

        st.divider()
        st.markdown("### Selected features")
        for i, feat in enumerate(BEST_FEATURES, 1):
            st.write(f"{i}. {feat.replace('_', ' ').title()}")

        st.divider()
        st.markdown("### Links")
        st.link_button("View on Hugging Face", f"https://huggingface.co/{HF_REPO_ID}", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_inputs() -> dict:
    # Session defaults (for paired controls)
    st.session_state.setdefault("satisfaction_level", 0.50)
    st.session_state.setdefault("last_evaluation", 0.70)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Employee inputs")
    st.markdown('<div class="hint">Use the sliders for quick adjustments and the numeric boxes for precise entry.</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2, vertical_alignment="top")
    with r1c1:
        satisfaction_level = paired_float_control(
            title="Satisfaction level",
            state_key="satisfaction_level",
            slider_key="sat_slider",
            input_key="sat_input",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help_text="Employee satisfaction level (0 = very dissatisfied, 1 = very satisfied).",
        )
    with r1c2:
        last_evaluation = paired_float_control(
            title="Last evaluation score",
            state_key="last_evaluation",
            slider_key="eval_slider",
            input_key="eval_input",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help_text="Last performance evaluation score (0 = poor, 1 = excellent).",
        )

    r2c1, r2c2 = st.columns(2, vertical_alignment="top")
    with r2c1:
        time_spend_company = st.number_input(
            "Years at company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            help="Number of years the employee has been with the company.",
        )
    with r2c2:
        number_project = st.number_input(
            "Number of projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of projects the employee is currently working on.",
        )

    r3c1, r3c2 = st.columns([1, 1], vertical_alignment="top")
    with r3c1:
        average_monthly_hours = st.number_input(
            "Average monthly hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            help="Average number of hours worked per month.",
        )
    with r3c2:
        st.caption(
            "Tip: Extreme workloads + low satisfaction often increase turnover risk. "
            "Use the probabilities below as a guide, not a final decision."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "satisfaction_level": float(satisfaction_level),
        "time_spend_company": int(time_spend_company),
        "average_monthly_hours": int(average_monthly_hours),
        "number_project": int(number_project),
        "last_evaluation": float(last_evaluation),
    }


def render_predict_cta() -> bool:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Predict turnover risk")
    st.markdown('<div class="hint">Click to generate a prediction with probabilities.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        clicked = st.button("Predict Employee Turnover", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


def render_results(prediction: int, prob_stay: float, prob_leave: float) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Results")

    left, right = st.columns([1.05, 1], vertical_alignment="top")

    with left:
        if prediction == 0:
            st.markdown(
                f"""
<div class="card result-good" style="box-shadow:none;">
  <div class="result-title">Stay</div>
  <p class="result-sub">The model estimates the employee is more likely to <b>stay</b>.</p>
</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="card result-bad" style="box-shadow:none;">
  <div class="result-title">Leave</div>
  <p class="result-sub">The model estimates the employee is more likely to <b>leave</b>.</p>
</div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("**Prediction probabilities**")
        st.write(f"Stay: **{prob_stay:.1f}%**")
        st.markdown(
            f"""
<div class="bar-wrap"><div class="bar bar-good" style="width:{prob_stay:.1f}%;"></div></div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"Leave: **{prob_leave:.1f}%**")
        st.markdown(
            f"""
<div class="bar-wrap"><div class="bar bar-bad" style="width:{prob_leave:.1f}%;"></div></div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_input_summary(input_data: dict) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.expander("View input summary", expanded=False):
        summary_df = pd.DataFrame(
            {
                "Feature": [
                    "Satisfaction level",
                    "Last evaluation",
                    "Years at company",
                    "Number of projects",
                    "Average monthly hours",
                ],
                "Value": [
                    f"{input_data['satisfaction_level']:.2f}",
                    f"{input_data['last_evaluation']:.2f}",
                    f"{input_data['time_spend_company']} years",
                    f"{input_data['number_project']} projects",
                    f"{input_data['average_monthly_hours']} hours",
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()
    render_hero()
    render_sidebar()

    model = load_model_from_huggingface()
    if model is None:
        st.error("Failed to load model. Check the Hugging Face repository and filename.")
        st.info(f"https://huggingface.co/{HF_REPO_ID}")
        return

    st.markdown("")  # spacer
    input_data = render_inputs()
    clicked = render_predict_cta()

    if clicked:
        # Predict with a visible, professional loading affordance
        with st.spinner("Running prediction…"):
            input_df = pd.DataFrame([input_data])[BEST_FEATURES]
            prediction = int(model.predict(input_df)[0])
            proba = model.predict_proba(input_df)[0]
            prob_stay = float(proba[0]) * 100.0
            prob_leave = float(proba[1]) * 100.0

        render_results(prediction, prob_stay, prob_leave)
        render_input_summary(input_data)


if __name__ == "__main__":
    main()
