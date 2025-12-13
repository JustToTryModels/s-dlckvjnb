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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg0: #0B1220;
  --bg1: #0F172A;
  --card: rgba(255,255,255,0.86);
  --card2: rgba(255,255,255,0.72);
  --stroke: rgba(15, 23, 42, 0.10);

  --text: #0F172A;
  --muted: rgba(15,23,42,0.68);

  --primary: #2563EB;  /* blue */
  --accent:  #14B8A6;  /* teal */
  --good:    #16A34A;  /* green */
  --bad:     #DC2626;  /* red */
  --warn:    #F59E0B;  /* amber */

  --shadow:  0 12px 40px rgba(2,6,23,0.18);
  --shadow2: 0 8px 26px rgba(2,6,23,0.14);
}

* { font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

/* App background */
div[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1000px 420px at 15% 0%, rgba(37,99,235,0.20), transparent 60%),
    radial-gradient(1000px 420px at 85% 0%, rgba(20,184,166,0.18), transparent 60%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

/* Main container padding */
.main .block-container{
  padding-top: 2.2rem;
  padding-bottom: 2.5rem;
  max-width: 1200px;
}

/* Sidebar styling */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.84));
  border-right: 1px solid rgba(15,23,42,0.08);
}
section[data-testid="stSidebar"] .block-container{
  padding-top: 1.5rem;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
  color: var(--text);
}

/* Hero header */
.hero{
  background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,255,255,0.74));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 1.4rem 1.5rem;
  margin-bottom: 1.2rem;
  position: relative;
  overflow: hidden;
  animation: enterHero 550ms ease-out both;
}
.hero:before{
  content: "";
  position: absolute;
  inset: -1px;
  background: radial-gradient(500px 200px at 30% 0%, rgba(37,99,235,0.18), transparent 60%),
              radial-gradient(500px 200px at 70% 0%, rgba(20,184,166,0.16), transparent 60%);
  pointer-events: none;
}
@keyframes enterHero{
  from{ transform: translateY(10px); opacity: 0; }
  to  { transform: translateY(0); opacity: 1; }
}
.hero-inner{
  position: relative;
  display: flex;
  gap: 14px;
  align-items: center;
}
.brand-mark{
  width: 44px; height: 44px;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(37,99,235,0.95), rgba(20,184,166,0.95));
  box-shadow: 0 10px 22px rgba(37,99,235,0.24);
  display: grid;
  place-items: center;
  color: white;
  flex: 0 0 auto;
}
.hero-title{
  font-size: 1.55rem;
  font-weight: 800;
  color: var(--text);
  line-height: 1.15;
  margin: 0;
}
.hero-sub{
  margin: 0.15rem 0 0 0;
  color: var(--muted);
  font-size: 0.98rem;
}
.chips{
  margin-top: 0.65rem;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.chip{
  font-size: 0.82rem;
  color: rgba(15,23,42,0.82);
  border: 1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.75);
  padding: 0.35rem 0.55rem;
  border-radius: 999px;
}

/* Card components */
.card{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  box-shadow: var(--shadow2);
  padding: 1.1rem 1.1rem;
}
.card.soft{
  background: var(--card2);
  backdrop-filter: blur(10px);
}
.card-title{
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.6rem;
}
.helptext{
  color: var(--muted);
  font-size: 0.92rem;
}

/* Inputs: give widgets a cohesive look */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input{
  border-radius: 12px !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus{
  border-color: rgba(37,99,235,0.55) !important;
  box-shadow: 0 0 0 4px rgba(37,99,235,0.12) !important;
}

/* Sliders */
div[data-testid="stSlider"] [role="slider"]{
  box-shadow: 0 0 0 4px rgba(20,184,166,0.12);
}
div[data-testid="stSlider"]{
  padding: 0.25rem 0.25rem 0.1rem 0.25rem;
  border-radius: 14px;
  transition: transform 140ms ease, box-shadow 140ms ease;
}
div[data-testid="stSlider"]:hover{
  transform: translateY(-1px);
}

/* CTA button */
.stButton>button{
  width: 100%;
  border: 0 !important;
  border-radius: 999px !important;
  padding: 0.95rem 1.1rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.7px;
  color: white !important;
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
  box-shadow: 0 14px 35px rgba(37,99,235,0.28);
  position: relative;
  overflow: hidden;
  transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
}
.stButton>button:hover{
  transform: translateY(-2px);
  box-shadow: 0 18px 46px rgba(37,99,235,0.34);
  filter: brightness(1.02);
}
.stButton>button:active{
  transform: translateY(0px) scale(0.99);
  box-shadow: 0 10px 26px rgba(37,99,235,0.26);
}

/* Shimmer overlay on hover */
.stButton>button::before{
  content: "";
  position: absolute;
  top: 0; left: -120%;
  width: 70%;
  height: 100%;
  background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
  transform: skewX(-18deg);
  transition: left 600ms ease;
}
.stButton>button:hover::before{ left: 140%; }

/* Result cards */
.result{
  border-radius: 18px;
  padding: 1.1rem 1.1rem;
  border: 1px solid var(--stroke);
  box-shadow: var(--shadow2);
  background: rgba(255,255,255,0.88);
  animation: enterResult 420ms ease-out both;
}
@keyframes enterResult{
  from{ transform: translateY(10px); opacity: 0; }
  to  { transform: translateY(0); opacity: 1; }
}
.result.good{ border-left: 6px solid var(--good); }
.result.bad { border-left: 6px solid var(--bad); }
.result h2{ margin: 0; color: var(--text); }
.result p{ margin: 0.35rem 0 0 0; color: var(--muted); }

/* Progress bars */
.pb-wrap{
  width: 100%;
  height: 12px;
  border-radius: 999px;
  background: rgba(15,23,42,0.08);
  overflow: hidden;
  border: 1px solid rgba(15,23,42,0.06);
}
.pb{
  height: 100%;
  width: 0%;
  border-radius: 999px;
  transition: width 650ms cubic-bezier(.2,.8,.2,1);
}
.pb.good{ background: linear-gradient(90deg, rgba(22,163,74,0.95), rgba(34,197,94,0.95)); }
.pb.bad { background: linear-gradient(90deg, rgba(220,38,38,0.95), rgba(239,68,68,0.95)); }

/* Donut chart */
.donut{
  --p1: 50; /* stay percent */
  width: 150px;
  height: 150px;
  border-radius: 50%;
  background:
    conic-gradient(
      rgba(22,163,74,0.95) 0 calc(var(--p1) * 1%),
      rgba(220,38,38,0.95) 0 100%
    );
  display: grid;
  place-items: center;
  box-shadow: 0 16px 34px rgba(2,6,23,0.18);
  border: 1px solid rgba(255,255,255,0.65);
}
.donut::after{
  content: "";
  width: 108px;
  height: 108px;
  border-radius: 50%;
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(15,23,42,0.08);
  box-shadow: inset 0 10px 22px rgba(2,6,23,0.08);
}
.donut-label{
  position: absolute;
  text-align: center;
  width: 150px;
}
.donut-num{
  font-weight: 900;
  color: var(--text);
  font-size: 1.05rem;
}
.donut-sub{
  color: var(--muted);
  font-size: 0.85rem;
}

/* Expander styling */
div[data-testid="stExpander"] details summary{
  border-radius: 14px !important;
  padding: 0.85rem 1.0rem !important;
  background: rgba(255,255,255,0.82) !important;
  border: 1px solid rgba(15,23,42,0.10) !important;
  box-shadow: 0 10px 24px rgba(2,6,23,0.10);
}
div[data-testid="stExpander"] details[open] summary{
  border-bottom-left-radius: 0 !important;
  border-bottom-right-radius: 0 !important;
}
div[data-testid="stExpander"] details > div{
  border: 1px solid rgba(15,23,42,0.10) !important;
  border-top: none !important;
  border-bottom-left-radius: 14px !important;
  border-bottom-right-radius: 14px !important;
  background: rgba(255,255,255,0.86) !important;
}

/* Responsive adjustments */
@media (max-width: 768px){
  .hero{ padding: 1rem 1rem; }
  .hero-title{ font-size: 1.3rem; }
  .chips{ margin-top: 0.55rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model_from_huggingface():
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
        )
        return joblib.load(model_path)
    except Exception as e:
        return e  # return the exception to display nicely


# =============================================================================
# WIDGET SYNC (slider <-> number input)
# =============================================================================
def sync_sat_from_slider():
    st.session_state.satisfaction_level = st.session_state.sat_slider
    st.session_state.sat_input = st.session_state.sat_slider


def sync_sat_from_input():
    st.session_state.satisfaction_level = st.session_state.sat_input
    st.session_state.sat_slider = st.session_state.sat_input


def sync_eval_from_slider():
    st.session_state.last_evaluation = st.session_state.eval_slider
    st.session_state.eval_input = st.session_state.eval_slider


def sync_eval_from_input():
    st.session_state.last_evaluation = st.session_state.eval_input
    st.session_state.eval_slider = st.session_state.eval_input


# =============================================================================
# UI RENDERING
# =============================================================================
def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="hero-inner">
    <div class="brand-mark">
      <!-- Simple inline SVG icon -->
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M16 11c1.66 0 3-1.57 3-3.5S17.66 4 16 4s-3 1.57-3 3.5S14.34 11 16 11Z" fill="white" opacity="0.95"/>
        <path d="M8 11c1.66 0 3-1.57 3-3.5S9.66 4 8 4 5 5.57 5 7.5 6.34 11 8 11Z" fill="white" opacity="0.85"/>
        <path d="M8 13c-2.76 0-5 2.24-5 5v1h10v-1c0-2.76-2.24-5-5-5Z" fill="white" opacity="0.85"/>
        <path d="M16 13c-1.15 0-2.2.35-3.06.95 1.25 1.11 2.06 2.73 2.06 4.55v1h6v-1c0-2.76-2.24-5-5-5Z" fill="white" opacity="0.95"/>
      </svg>
    </div>
    <div>
      <p class="hero-title">Employee Turnover Prediction</p>
      <p class="hero-sub">Predict turnover risk with a lightweight ML model and a clean, transparent input summary.</p>
      <div class="chips">
        <span class="chip">Random Forest • Balanced classes</span>
        <span class="chip">5 key features</span>
        <span class="chip">Probability outputs</span>
      </div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Model")
        st.markdown(
            f"""
<div class="card soft">
  <div class="card-title">Repository</div>
  <div class="helptext"><a href="https://huggingface.co/{HF_REPO_ID}" target="_blank">huggingface.co/{HF_REPO_ID}</a></div>
  <hr style="border:none;border-top:1px solid rgba(15,23,42,0.10);margin:0.9rem 0;">
  <div class="card-title">Algorithm</div>
  <div class="helptext">Random Forest Classifier</div>
  <div style="height:10px"></div>
  <div class="card-title">Features used</div>
  <div class="helptext">{len(BEST_FEATURES)} selected features</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Selected features")
        st.markdown(
            "<div class='card soft'>"
            + "".join([f"<div style='margin:0.35rem 0; color:rgba(15,23,42,0.82)'>• {f.replace('_',' ').title()}</div>" for f in BEST_FEATURES])
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Target")
        st.markdown(
            """
<div class="card soft">
  <div class="helptext"><b>0 → Stay</b> (likely to remain)</div>
  <div class="helptext" style="margin-top:0.35rem;"><b>1 → Leave</b> (likely to exit)</div>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_inputs_form():
    # Defaults for synced fields
    st.session_state.setdefault("satisfaction_level", 0.50)
    st.session_state.setdefault("last_evaluation", 0.70)

    st.markdown(
        """
<div class="card">
  <div class="card-title">Employee inputs</div>
  <div class="helptext">Adjust the fields below, then click <b>Predict turnover risk</b>.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form("predict_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Satisfaction level**")
            s1, s2 = st.columns([3, 1])
            with s1:
                st.slider(
                    "Satisfaction",
                    0.0, 1.0,
                    value=float(st.session_state.satisfaction_level),
                    step=0.01,
                    help="Overall satisfaction (0 = very dissatisfied, 1 = very satisfied).",
                    label_visibility="collapsed",
                    key="sat_slider",
                    on_change=sync_sat_from_slider,
                )
            with s2:
                st.number_input(
                    "Satisfaction value",
                    0.0, 1.0,
                    value=float(st.session_state.satisfaction_level),
                    step=0.01,
                    format="%.2f",
                    help="Same as the slider; you can type an exact value.",
                    label_visibility="collapsed",
                    key="sat_input",
                    on_change=sync_sat_from_input,
                )

        with c2:
            st.markdown("**Last evaluation score**")
            e1, e2 = st.columns([3, 1])
            with e1:
                st.slider(
                    "Evaluation",
                    0.0, 1.0,
                    value=float(st.session_state.last_evaluation),
                    step=0.01,
                    help="Last performance evaluation (0 = poor, 1 = excellent).",
                    label_visibility="collapsed",
                    key="eval_slider",
                    on_change=sync_eval_from_slider,
                )
            with e2:
                st.number_input(
                    "Evaluation value",
                    0.0, 1.0,
                    value=float(st.session_state.last_evaluation),
                    step=0.01,
                    format="%.2f",
                    help="Same as the slider; you can type an exact value.",
                    label_visibility="collapsed",
                    key="eval_input",
                    on_change=sync_eval_from_input,
                )

        st.write("")
        r2c1, r2c2, r2c3 = st.columns([1, 1, 1])

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
                help="How many projects the employee is currently assigned to.",
            )
        with r2c3:
            average_monthly_hours = st.number_input(
                "Average monthly hours",
                min_value=80,
                max_value=350,
                value=200,
                step=5,
                help="Average working hours per month.",
            )

        st.write("")
        submit = st.form_submit_button("Predict turnover risk")

    input_data = {
        "satisfaction_level": float(st.session_state.satisfaction_level),
        "time_spend_company": int(time_spend_company),
        "average_monthly_hours": int(average_monthly_hours),
        "number_project": int(number_project),
        "last_evaluation": float(st.session_state.last_evaluation),
    }

    return submit, input_data


def render_results(prediction: int, prob_stay: float, prob_leave: float) -> None:
    # Ensure clean numeric formatting
    prob_stay = float(prob_stay)
    prob_leave = float(prob_leave)

    left, right = st.columns([1.35, 1])

    with left:
        if prediction == 0:
            st.markdown(
                f"""
<div class="result good">
  <h2>Likely outcome: Stay</h2>
  <p>Based on the provided inputs, the employee is more likely to remain with the company.</p>
</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="result bad">
  <h2>Likely outcome: Leave</h2>
  <p>Based on the provided inputs, the employee shows higher turnover risk.</p>
</div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown(
            """
<div class="card">
  <div class="card-title">Prediction probabilities</div>
  <div class="helptext" style="margin-bottom:0.65rem;">Smooth bars animate to the model outputs.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        st.markdown(f"**Stay:** {prob_stay:.1f}%")
        st.markdown(
            f"""
<div class="pb-wrap">
  <div class="pb good" style="width:{prob_stay:.1f}%"></div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        st.markdown(f"**Leave:** {prob_leave:.1f}%")
        st.markdown(
            f"""
<div class="pb-wrap">
  <div class="pb bad" style="width:{prob_leave:.1f}%"></div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
<div class="card">
  <div class="card-title">Probability ring</div>
  <div class="helptext">Green = Stay, Red = Leave</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        # Donut uses stay percent for the split.
        st.markdown(
            f"""
<div style="position:relative; display:flex; justify-content:center; padding:0.8rem 0;">
  <div class="donut" style="--p1:{prob_stay:.1f}"></div>
  <div class="donut-label">
    <div class="donut-num">{max(prob_stay, prob_leave):.1f}%</div>
    <div class="donut-sub">{'Stay' if prob_stay >= prob_leave else 'Leave'} confidence</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_input_summary(input_data: dict) -> None:
    with st.expander("Input summary", expanded=False):
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


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    inject_css()
    render_hero()
    render_sidebar()

    model_or_err = load_model_from_huggingface()
    if isinstance(model_or_err, Exception):
        st.error(f"Error loading model from Hugging Face: {model_or_err}")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return

    model = model_or_err

    submitted, input_data = render_inputs_form()

    if submitted:
        input_df = pd.DataFrame([input_data])[BEST_FEATURES]

        with st.spinner("Running prediction…"):
            prediction = int(model.predict(input_df)[0])
            proba = model.predict_proba(input_df)[0]
            prob_stay = float(proba[0]) * 100.0
            prob_leave = float(proba[1]) * 100.0

        st.write("")
        render_results(prediction, prob_stay, prob_leave)
        st.write("")
        render_input_summary(input_data)


if __name__ == "__main__":
    main()
