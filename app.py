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
# STYLES
# =============================================================================
def inject_global_css() -> None:
    st.markdown(
        """
<style>
/* =========================
   Fonts + Theme Variables
   ========================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg: #0B1220;           /* deep navy */
  --bg2: #0E1A2F;
  --surface: rgba(255,255,255,0.06);
  --surface2: rgba(255,255,255,0.09);
  --card: rgba(255,255,255,0.08);
  --stroke: rgba(255,255,255,0.12);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);

  --accent: #21C7A8;       /* teal */
  --accent2: #2F80ED;      /* blue */
  --good: #2ECC71;
  --bad: #FF5A6E;
  --warn: #FFC857;

  --shadow: 0 16px 40px rgba(0,0,0,0.35);
  --shadow2: 0 12px 26px rgba(0,0,0,0.28);
  --radius-xl: 22px;
  --radius-lg: 16px;
  --radius-md: 12px;
}

html, body, [class*="css"]  {
  font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
}

/* =========================
   App Background + Layout
   ========================= */
.stApp {
  background:
    radial-gradient(1200px 800px at 15% 10%, rgba(47,128,237,0.25), transparent 60%),
    radial-gradient(1000px 700px at 85% 20%, rgba(33,199,168,0.18), transparent 65%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
  color: var(--text);
}

section.main > div { padding-top: 1.2rem; }

/* Hide Streamlit default footer/menu for a more product-like look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* =========================
   Sidebar Styling
   ========================= */
[data-testid="stSidebar"] > div:first-child {
  background:
    radial-gradient(700px 400px at 20% 10%, rgba(33,199,168,0.10), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.03) 100%);
  border-right: 1px solid var(--stroke);
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,  [data-testid="stSidebar"] li, [data-testid="stSidebar"] span {
  color: var(--text) !important;
}

.sidebar-card {
  background: rgba(0,0,0,0.18);
  border: 1px solid var(--stroke);
  border-radius: var(--radius-lg);
  padding: 14px 14px;
  box-shadow: var(--shadow2);
}

/* =========================
   Hero Header
   ========================= */
.hero {
  position: relative;
  padding: 22px 22px;
  border-radius: var(--radius-xl);
  border: 1px solid var(--stroke);
  background:
    linear-gradient(135deg, rgba(47,128,237,0.20), rgba(33,199,168,0.12)),
    rgba(255,255,255,0.05);
  box-shadow: var(--shadow);
  overflow: hidden;
  animation: fadeUp 520ms ease-out both;
}

.hero::before{
  content:"";
  position:absolute;
  inset:-2px;
  background:
    radial-gradient(600px 220px at 10% 0%, rgba(33,199,168,0.22), transparent 65%),
    radial-gradient(520px 240px at 90% 10%, rgba(47,128,237,0.22), transparent 60%);
  filter: blur(18px);
  opacity: 0.75;
  pointer-events:none;
}

.hero-grid {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 16px;
  position: relative;
  z-index: 1;
}

.hero-title {
  font-size: 2.0rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0;
  color: var(--text);
}

.hero-subtitle {
  margin-top: 8px;
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.5;
}

.badge-row { display:flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.badge {
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--stroke);
  background: rgba(0,0,0,0.18);
  color: var(--text);
  font-size: 0.9rem;
}

.hero-icon {
  width: 56px; height: 56px;
  border-radius: 16px;
  display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, rgba(33,199,168,0.35), rgba(47,128,237,0.25));
  border: 1px solid rgba(255,255,255,0.16);
  box-shadow: 0 10px 25px rgba(0,0,0,0.22);
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0px); }
}

/* =========================
   Cards + Section Titles
   ========================= */
.section-title {
  margin: 18px 0 10px 0;
  font-size: 1.15rem;
  font-weight: 800;
  letter-spacing: -0.01em;
}

.card {
  border-radius: var(--radius-xl);
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.06);
  box-shadow: var(--shadow2);
  padding: 18px 18px;
}

.card-soft {
  border-radius: var(--radius-xl);
  border: 1px solid var(--stroke);
  background: rgba(0,0,0,0.18);
  box-shadow: var(--shadow2);
  padding: 18px 18px;
}

/* =========================
   Inputs polish (BaseWeb)
   ========================= */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
  background: rgba(0,0,0,0.25) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 12px !important;
}

div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
  border-color: rgba(33,199,168,0.55) !important;
  box-shadow: 0 0 0 4px rgba(33,199,168,0.18) !important;
}

/* Slider track + thumb */
div[data-testid="stSlider"] > div {
  padding: 6px 8px 0px 8px;
  border-radius: 14px;
  background: rgba(0,0,0,0.16);
  border: 1px solid rgba(255,255,255,0.10);
}
div[data-baseweb="slider"] div[role="slider"] {
  box-shadow: 0 0 0 4px rgba(47,128,237,0.15);
}
div[data-baseweb="slider"] div[role="slider"]:hover {
  transform: scale(1.02);
  transition: transform 120ms ease;
}

/* =========================
   CTA Button (Professional)
   ========================= */
.stButton > button {
  width: 100%;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 999px !important;
  padding: 0.95rem 1.2rem !important;

  color: white !important;
  font-weight: 800 !important;
  letter-spacing: 0.02em !important;
  font-size: 1.05rem !important;

  background: linear-gradient(135deg, rgba(47,128,237,0.95), rgba(33,199,168,0.95)) !important;
  box-shadow: 0 14px 36px rgba(0,0,0,0.35);
  position: relative;
  overflow: hidden;
  transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
}

.stButton > button::before{
  content:"";
  position:absolute;
  top:0; left:-120%;
  width: 120%;
  height: 100%;
  background: linear-gradient(110deg, transparent, rgba(255,255,255,0.22), transparent);
  transform: skewX(-18deg);
}

.stButton > button:hover {
  transform: translateY(-2px);
  filter: saturate(1.05);
  box-shadow: 0 18px 44px rgba(0,0,0,0.42);
}

.stButton > button:hover::before{
  left: 120%;
  transition: left 620ms ease;
}

.stButton > button:active{
  transform: translateY(0px) scale(0.99);
}

/* =========================
   Result Cards
   ========================= */
.result-wrap { animation: fadeUp 420ms ease-out both; }

.result-card {
  border-radius: var(--radius-xl);
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.06);
  box-shadow: var(--shadow2);
  padding: 18px 18px;
}

.result-title {
  font-size: 1.2rem;
  font-weight: 900;
  margin: 0;
  letter-spacing: -0.01em;
}

.result-pill {
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--stroke);
  background: rgba(0,0,0,0.18);
  color: var(--text);
  font-weight: 700;
}

.result-stay { border-color: rgba(46,204,113,0.35); }
.result-leave{ border-color: rgba(255,90,110,0.35); }

.kpi {
  display:flex;
  justify-content:space-between;
  align-items:baseline;
  gap: 10px;
  margin: 10px 0 6px 0;
  color: var(--muted);
}

.progress-shell {
  width: 100%;
  height: 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  overflow:hidden;
}

.progress-bar {
  height: 100%;
  width: var(--w);
  border-radius: 999px;
  transition: width 650ms cubic-bezier(.2,.9,.2,1);
}

.progress-green { background: linear-gradient(90deg, rgba(46,204,113,0.85), rgba(33,199,168,0.85)); }
.progress-red   { background: linear-gradient(90deg, rgba(255,90,110,0.92), rgba(255,200,87,0.70)); }

/* =========================
   Expander styling
   ========================= */
div[data-testid="stExpander"] details summary {
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: 14px !important;
  padding: 0.85rem 1rem !important;
  font-weight: 800 !important;
}

div[data-testid="stExpander"] details > div {
  border: 1px solid var(--stroke) !important;
  border-top: none !important;
  border-radius: 0 0 14px 14px !important;
  background: rgba(0,0,0,0.18) !important;
}

/* =========================
   Responsive tweaks
   ========================= */
@media (max-width: 900px){
  .hero-title { font-size: 1.55rem; }
  .hero-grid { flex-direction: column; align-items:flex-start; }
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
# Slider/Input Sync
# =============================================================================
def sync_satisfaction_slider():
    st.session_state.satisfaction_level = st.session_state.sat_slider


def sync_satisfaction_input():
    st.session_state.satisfaction_level = st.session_state.sat_input


def sync_evaluation_slider():
    st.session_state.last_evaluation = st.session_state.eval_slider


def sync_evaluation_input():
    st.session_state.last_evaluation = st.session_state.eval_input


# =============================================================================
# UI RENDERING
# =============================================================================
def render_hero():
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-grid">
    <div>
      <div style="display:flex; align-items:center; gap:12px;">
        <div class="hero-icon" aria-hidden="true">
          <span style="font-size:28px;">👥</span>
        </div>
        <div>
          <h1 class="hero-title">Employee Turnover Prediction</h1>
          <div class="hero-subtitle">Enter a few key signals and get a clear, probability-based turnover risk estimate.</div>
        </div>
      </div>

      <div class="badge-row">
        <span class="badge">🤖 Random Forest</span>
        <span class="badge">⚖️ Balanced Classes</span>
        <span class="badge">🧩 {len(BEST_FEATURES)} Features</span>
      </div>
    </div>

    <div class="badge" style="background: rgba(0,0,0,0.22);">
      <span style="opacity:0.85;">Repo</span>
      <span style="font-weight:800;">{HF_REPO_ID}</span>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""
<div class="sidebar-card">
  <div style="font-weight:900; font-size:1.05rem; margin-bottom:10px;">Model Overview</div>
  <div style="color:rgba(255,255,255,0.80); line-height:1.55;">
    <div><span style="opacity:0.75;">Repository:</span><br><b>{HF_REPO_ID}</b></div>
    <div style="margin-top:10px;"><span style="opacity:0.75;">Algorithm:</span><br><b>Random Forest Classifier</b></div>
    <div style="margin-top:10px;"><span style="opacity:0.75;">Features:</span><br><b>{len(BEST_FEATURES)} selected signals</b></div>
    <div style="margin-top:10px;"><span style="opacity:0.75;">Class Weight:</span><br><b>Balanced</b></div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        st.markdown(
            """
<div class="sidebar-card">
  <div style="font-weight:900; font-size:1.05rem; margin-bottom:10px;">Selected Features</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        for i, feat in enumerate(BEST_FEATURES, 1):
            st.write(f"{i}. {feat.replace('_', ' ').title()}")

        st.markdown("")
        st.markdown(
            """
<div class="sidebar-card">
  <div style="font-weight:900; font-size:1.05rem; margin-bottom:10px;">Target Variable</div>
  <div style="color:rgba(255,255,255,0.78); line-height:1.55;">
    <b>0 → Stay</b>: likely to remain<br/>
    <b>1 → Leave</b>: likely to exit
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown(f"[View on Hugging Face ↗](https://huggingface.co/{HF_REPO_ID})")


def render_inputs() -> dict:
    st.markdown('<div class="section-title">Employee Signals</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-soft">
  <div style="font-weight:800; margin-bottom:6px;">How to use</div>
  <div style="color:rgba(255,255,255,0.78); line-height:1.55;">
    Adjust sliders or type values. Then click <b>Predict</b> to see a probability split for staying vs leaving.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Defaults for sync values
    if "satisfaction_level" not in st.session_state:
        st.session_state.satisfaction_level = 0.50
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = 0.70

    # Row 1
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**Satisfaction Level**")
        sc1, sc2 = st.columns([3, 1], gap="small")
        with sc1:
            st.slider(
                "Satisfaction Slider",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.satisfaction_level),
                step=0.01,
                help="Overall satisfaction (0 = very dissatisfied, 1 = very satisfied).",
                label_visibility="collapsed",
                key="sat_slider",
                on_change=sync_satisfaction_slider,
            )
        with sc2:
            st.number_input(
                "Satisfaction Input",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.satisfaction_level),
                step=0.01,
                format="%.2f",
                help="You can type a precise value here.",
                label_visibility="collapsed",
                key="sat_input",
                on_change=sync_satisfaction_input,
            )
        satisfaction_level = float(st.session_state.satisfaction_level)

    with c2:
        st.markdown("**Last Evaluation Score**")
        ec1, ec2 = st.columns([3, 1], gap="small")
        with ec1:
            st.slider(
                "Evaluation Slider",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.last_evaluation),
                step=0.01,
                help="Most recent performance evaluation score (0 = poor, 1 = excellent).",
                label_visibility="collapsed",
                key="eval_slider",
                on_change=sync_evaluation_slider,
            )
        with ec2:
            st.number_input(
                "Evaluation Input",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.last_evaluation),
                step=0.01,
                format="%.2f",
                help="You can type a precise value here.",
                label_visibility="collapsed",
                key="eval_input",
                on_change=sync_evaluation_input,
            )
        last_evaluation = float(st.session_state.last_evaluation)

    # Row 2
    c3, c4 = st.columns(2, gap="large")
    with c3:
        time_spend_company = st.number_input(
            "Years at Company",
            min_value=1,
            max_value=40,
            value=3,
            step=1,
            help="Number of years the employee has worked at the company.",
        )
    with c4:
        number_project = st.number_input(
            "Number of Projects",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="How many projects the employee is currently assigned to.",
        )

    # Row 3
    c5, _ = st.columns(2, gap="large")
    with c5:
        average_monthly_hours = st.number_input(
            "Average Monthly Hours",
            min_value=80,
            max_value=350,
            value=200,
            step=5,
            help="Average number of hours worked per month.",
        )

    # If user changes inputs, clear previous result (keeps UX clean)
    current_signature = (
        satisfaction_level,
        last_evaluation,
        int(time_spend_company),
        int(number_project),
        int(average_monthly_hours),
    )
    if st.session_state.get("last_signature") != current_signature:
        st.session_state.last_signature = current_signature
        st.session_state.last_result = None

    return {
        "satisfaction_level": satisfaction_level,
        "time_spend_company": int(time_spend_company),
        "average_monthly_hours": int(average_monthly_hours),
        "number_project": int(number_project),
        "last_evaluation": last_evaluation,
    }


def render_results(input_data: dict):
    result = st.session_state.get("last_result")
    if not result:
        return

    prediction = result["prediction"]
    prob_stay = result["prob_stay"]
    prob_leave = result["prob_leave"]

    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1.2], gap="large")
    with left:
        if prediction == 0:
            st.markdown(
                f"""
<div class="result-wrap">
  <div class="result-card result-stay">
    <div class="result-pill">✅ Outcome</div>
    <div style="margin-top:10px;">
      <p class="result-title">Likely to Stay</p>
      <div style="color: rgba(255,255,255,0.78); line-height:1.55; margin-top:8px;">
        Based on the provided signals, the model leans toward <b>retention</b>.
      </div>
    </div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="result-wrap">
  <div class="result-card result-leave">
    <div class="result-pill">⚠️ Outcome</div>
    <div style="margin-top:10px;">
      <p class="result-title">Likely to Leave</p>
      <div style="color: rgba(255,255,255,0.78); line-height:1.55; margin-top:8px;">
        Based on the provided signals, the model leans toward <b>turnover risk</b>.
      </div>
    </div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(
            """
<div class="result-wrap">
  <div class="result-card">
    <div class="result-pill">📊 Probabilities</div>
    <div style="margin-top:12px;">
      <div class="kpi"><span><b>Stay</b></span><span style="color:rgba(255,255,255,0.90); font-weight:900;">{stay:.1f}%</span></div>
      <div class="progress-shell"><div class="progress-bar progress-green" style="--w: {stay:.2f}%;"></div></div>

      <div style="height:12px;"></div>

      <div class="kpi"><span><b>Leave</b></span><span style="color:rgba(255,255,255,0.90); font-weight:900;">{leave:.1f}%</span></div>
      <div class="progress-shell"><div class="progress-bar progress-red" style="--w: {leave:.2f}%;"></div></div>

      <div style="margin-top:14px; color:rgba(255,255,255,0.70); line-height:1.55;">
        Tip: treat this as a decision support signal. Consider recent role changes, manager context, and HR policy before acting.
      </div>
    </div>
  </div>
</div>
            """.format(
                stay=prob_stay, leave=prob_leave
            ),
            unsafe_allow_html=True,
        )

    # Input Summary
    st.markdown("")
    with st.expander("Input Summary", expanded=False):
        # Clean HTML table (more “website-like” than default dataframe)
        rows = [
            ("😊 Satisfaction Level", f"{input_data['satisfaction_level']:.2f}"),
            ("📊 Last Evaluation", f"{input_data['last_evaluation']:.2f}"),
            ("📅 Years at Company", f"{input_data['time_spend_company']}"),
            ("📁 Number of Projects", f"{input_data['number_project']}"),
            ("⏰ Average Monthly Hours", f"{input_data['average_monthly_hours']}"),
        ]
        table_rows = "\n".join(
            [f"<tr><td style='padding:10px 8px; opacity:0.9;'><b>{k}</b></td><td style='padding:10px 8px; text-align:right; font-weight:800;'>{v}</td></tr>" for k, v in rows]
        )
        st.markdown(
            f"""
<div style="border:1px solid rgba(255,255,255,0.12); border-radius:14px; overflow:hidden;">
  <table style="width:100%; border-collapse:collapse; background: rgba(255,255,255,0.04);">
    <tbody>
      {table_rows}
    </tbody>
  </table>
</div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_global_css()
    render_sidebar()
    render_hero()

    model = load_model_from_huggingface()
    if model is None:
        st.error("Failed to load the model from Hugging Face.")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return

    input_data = render_inputs()

    st.markdown("")
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        predict_button = st.button("Predict Turnover Risk", use_container_width=True)

    if predict_button:
        # Visually polished loading flow
        with st.spinner("Running model inference..."):
            input_df = pd.DataFrame([input_data])[BEST_FEATURES]
            pred = int(model.predict(input_df)[0])
            proba = model.predict_proba(input_df)[0]

        st.session_state.last_result = {
            "prediction": pred,
            "prob_stay": float(proba[0] * 100.0),
            "prob_leave": float(proba[1] * 100.0),
        }

    render_results(input_data)


if __name__ == "__main__":
    main()
