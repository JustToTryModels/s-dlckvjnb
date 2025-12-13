import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# Optional (recommended) for gauge/radial visuals
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONFIG
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
# THEME + PREMIUM CSS
# =============================================================================
def inject_premium_css(theme: str) -> None:
    """
    Premium UI layer: gradients, glassmorphism, hover interactions, responsive tweaks.
    Uses two modes: 'dark' (default) and 'light'.
    """
    dark = (theme == "dark")

    # Palette requested by user prompt
    NAVY = "#0F1C3F"
    ACCENT = "#2E7D9E"
    SUCCESS = "#10B981"
    WARNING = "#F59E0B"
    DANGER = "#EF4444"

    BG0 = "#070B18" if dark else "#F6F8FC"
    BG1 = NAVY if dark else "#FFFFFF"
    TEXT = "#EAF0FF" if dark else "#0B1220"
    MUTED = "#A9B4D0" if dark else "#52637A"
    CARD = "rgba(255,255,255,0.06)" if dark else "rgba(15,28,63,0.04)"
    CARD_BORDER = "rgba(255,255,255,0.14)" if dark else "rgba(15,28,63,0.12)"

    st.markdown(
        f"""
<style>
/* ===== Root tokens ===== */
:root {{
  --navy: {NAVY};
  --accent: {ACCENT};
  --success: {SUCCESS};
  --warning: {WARNING};
  --danger: {DANGER};

  --bg0: {BG0};
  --bg1: {BG1};
  --text: {TEXT};
  --muted: {MUTED};

  --card: {CARD};
  --card-border: {CARD_BORDER};
  --shadow: 0 20px 50px rgba(0,0,0,0.35);
  --shadow-soft: 0 12px 30px rgba(0,0,0,0.22);
  --radius-lg: 18px;
  --radius-md: 14px;
  --radius-sm: 12px;

  --grad-hero: linear-gradient(135deg, rgba(46,125,158,0.95), rgba(15,28,63,0.95));
  --grad-accent: linear-gradient(135deg, rgba(46,125,158,1), rgba(16,185,129,1));
  --grad-danger: linear-gradient(135deg, rgba(239,68,68,1), rgba(245,158,11,1));
  --grad-line: linear-gradient(90deg, rgba(46,125,158,0), rgba(46,125,158,0.85), rgba(46,125,158,0));
}}

/* ===== App background ===== */
.stApp {{
  background:
    radial-gradient(1200px 600px at 20% -10%, rgba(46,125,158,0.35), rgba(0,0,0,0) 60%),
    radial-gradient(1000px 600px at 95% 10%, rgba(16,185,129,0.22), rgba(0,0,0,0) 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text);
}}

/* Subtle animated background sheen */
@keyframes bgShift {{
  0% {{ filter: hue-rotate(0deg); }}
  50% {{ filter: hue-rotate(6deg); }}
  100% {{ filter: hue-rotate(0deg); }}
}}
.stApp::before {{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events: none;
  animation: bgShift 10s ease-in-out infinite;
}}

/* ===== Typography ===== */
html, body, [class*="css"] {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}}
h1, h2, h3 {{
  letter-spacing: -0.02em;
}}
.small-muted {{
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.35;
}}

/* ===== Gradient separators ===== */
.hr-grad {{
  height: 1px;
  width: 100%;
  background: var(--grad-line);
  border: 0;
  margin: 18px 0 18px 0;
  opacity: 0.9;
}}

/* ===== Glass cards ===== */
.card {{
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: var(--radius-lg);
  padding: 18px 18px;
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
}}
.card:hover {{
  transform: translateY(-2px) scale(1.01);
  box-shadow: var(--shadow);
  border-color: rgba(46,125,158,0.55);
}}
.cardTitle {{
  font-weight: 700;
  font-size: 1.05rem;
  margin-bottom: 6px;
}}
.cardKPI {{
  font-weight: 800;
  font-size: 2rem;
  line-height: 1.05;
}}
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 999px;
  font-weight: 700;
  border: 1px solid var(--card-border);
  background: rgba(255,255,255,0.06);
}}
.badge.good {{
  border-color: rgba(16,185,129,0.55);
}}
.badge.warn {{
  border-color: rgba(245,158,11,0.55);
}}
.badge.bad {{
  border-color: rgba(239,68,68,0.55);
}}

/* ===== Hero header ===== */
.hero {{
  background: var(--grad-hero);
  border-radius: 22px;
  padding: 22px 22px;
  border: 1px solid rgba(255,255,255,0.16);
  box-shadow: var(--shadow-soft);
  position: relative;
  overflow: hidden;
}}
.hero::after {{
  content:"";
  position:absolute;
  top:-30%;
  left:-30%;
  width: 60%;
  height: 160%;
  background: radial-gradient(circle at 30% 40%, rgba(255,255,255,0.18), rgba(255,255,255,0) 60%);
  transform: rotate(18deg);
}}
.heroTitle {{
  font-size: 2.2rem;
  font-weight: 900;
  margin: 0 0 6px 0;
  text-shadow: 0 10px 30px rgba(0,0,0,0.35);
}}
.heroSub {{
  font-size: 1.05rem;
  margin: 0;
  color: rgba(255,255,255,0.86);
}}

/* ===== Animations ===== */
@keyframes riseIn {{
  from {{ opacity: 0; transform: translateY(10px); }}
  to {{ opacity: 1; transform: translateY(0px); }}
}}
.riseIn {{
  animation: riseIn 450ms ease both;
}}

/* ===== Buttons (premium, less “rainbow”, more brand-consistent) ===== */
.stButton>button {{
  width: 100%;
  border: 0 !important;
  border-radius: 999px !important;
  padding: 0.95rem 1.2rem !important;
  font-weight: 900 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: white !important;
  background: var(--grad-accent) !important;
  box-shadow: 0 16px 40px rgba(46,125,158,0.35), 0 8px 24px rgba(16,185,129,0.18);
  transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
  position: relative;
  overflow: hidden;
}}
.stButton>button:hover {{
  transform: translateY(-2px);
  filter: brightness(1.04);
  box-shadow: 0 22px 55px rgba(46,125,158,0.45), 0 10px 30px rgba(16,185,129,0.22);
}}
.stButton>button:active {{
  transform: translateY(1px) scale(0.99);
}}
/* Ripple-like sheen */
.stButton>button::before {{
  content:"";
  position:absolute;
  inset:-40%;
  background: linear-gradient(120deg, rgba(255,255,255,0), rgba(255,255,255,0.25), rgba(255,255,255,0));
  transform: translateX(-60%) rotate(12deg);
  transition: transform 420ms ease;
}}
.stButton>button:hover::before {{
  transform: translateX(60%) rotate(12deg);
}}
/* Kill focus outlines while keeping a visible focus shadow */
.stButton>button:focus, .stButton>button:focus-visible {{
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(46,125,158,0.35), 0 22px 55px rgba(46,125,158,0.45);
}}

/* ===== Inputs: subtle focus glow ===== */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {{
  border-radius: 12px !important;
}}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {{
  box-shadow: 0 0 0 3px rgba(46,125,158,0.28) !important;
  border-color: rgba(46,125,158,0.65) !important;
}}
/* Slider container */
div[data-testid="stSlider"] > div {{
  padding: 6px 10px;
  border-radius: 14px;
  border: 1px solid var(--card-border);
  background: rgba(255,255,255,0.04);
}}

/* ===== Expander styling ===== */
div[data-testid="stExpander"] details summary {{
  border-radius: 14px !important;
  padding: 0.85rem 1rem !important;
  border: 1px solid var(--card-border) !important;
  background: rgba(255,255,255,0.06) !important;
}}
div[data-testid="stExpander"] details[open] summary {{
  border-bottom-left-radius: 0 !important;
  border-bottom-right-radius: 0 !important;
}}

/* ===== Sidebar: glass panel ===== */
section[data-testid="stSidebar"] > div {{
  background:
    radial-gradient(600px 300px at 30% 0%, rgba(46,125,158,0.18), rgba(0,0,0,0) 55%),
    rgba(255,255,255,0.04);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255,255,255,0.12);
}}

/* ===== Responsive tweaks ===== */
@media (max-width: 900px) {{
  .heroTitle {{ font-size: 1.7rem; }}
  .cardKPI {{ font-size: 1.6rem; }}
}}
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
        return e  # return exception so we can display it cleanly


# =============================================================================
# SMALL HELPERS
# =============================================================================
def make_gauge(prob_leave_pct: float):
    """Plotly gauge. Fallback handled elsewhere."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_leave_pct,
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2E7D9E"},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.14)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(16,185,129,0.35)"},
                    {"range": [35, 65], "color": "rgba(245,158,11,0.35)"},
                    {"range": [65, 100], "color": "rgba(239,68,68,0.35)"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
            title={"text": "Leave Risk", "font": {"size": 16}},
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=10),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def recommendations(input_data: dict, prob_leave: float) -> list[str]:
    """
    Non-causal, guidance-oriented suggestions based on high-level signals.
    """
    recs: list[str] = []
    if prob_leave >= 0.65:
        recs.append("Prioritize a retention check-in: manager 1:1 + role clarity + growth plan.")
    if input_data["satisfaction_level"] <= 0.40:
        recs.append("Investigate satisfaction drivers: workload, recognition, autonomy, team fit.")
    if input_data["average_monthly_hours"] >= 240:
        recs.append("Review workload: rebalance projects, reduce overtime, protect focus time.")
    if input_data["number_project"] >= 6:
        recs.append("Reduce context switching: limit concurrent projects or redefine priorities.")
    if input_data["time_spend_company"] >= 4 and input_data["last_evaluation"] >= 0.75:
        recs.append("High performer signal: ensure compensation alignment and advancement path.")
    if not recs:
        recs.append("Maintain engagement: regular feedback, growth opportunities, balanced workload.")
    return recs


# =============================================================================
# APP
# =============================================================================
def main():
    # Theme toggle (CSS-only theming inside app)
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"

    inject_premium_css(st.session_state.theme_mode)

    # HERO
    st.markdown(
        """
        <div class="hero riseIn">
          <div class="heroTitle">Employee Turnover Prediction</div>
          <p class="heroSub">Predict whether an employee is likely to leave using a Random Forest model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hr-grad"></div>', unsafe_allow_html=True)

    # Load model
    model_or_exc = load_model_from_huggingface()
    if isinstance(model_or_exc, Exception):
        st.error(f"Error loading model: {model_or_exc}")
        st.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return
    model = model_or_exc

    # SIDEBAR
    with st.sidebar:
        st.markdown("### Settings")
        theme_choice = st.toggle("Light mode", value=(st.session_state.theme_mode == "light"))
        st.session_state.theme_mode = "light" if theme_choice else "dark"
        inject_premium_css(st.session_state.theme_mode)

        st.markdown('<div class="hr-grad"></div>', unsafe_allow_html=True)

        st.markdown("### Model Information")
        st.markdown(
            f"""
<div class="card">
  <div class="cardTitle">Repository</div>
  <div class="small-muted">{HF_REPO_ID}</div>
  <div style="height:10px"></div>
  <div class="cardTitle">Algorithm</div>
  <div class="small-muted">Random Forest Classifier (class_weight=balanced)</div>
  <div style="height:10px"></div>
  <div class="cardTitle">Features</div>
  <div class="small-muted">{len(BEST_FEATURES)} selected features</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="hr-grad"></div>', unsafe_allow_html=True)
        st.markdown("### Features Used")
        for i, feat in enumerate(BEST_FEATURES, 1):
            st.write(f"{i}. {feat.replace('_', ' ').title()}")

        st.markdown('<div class="hr-grad"></div>', unsafe_allow_html=True)
        st.markdown("[View on Hugging Face](https://huggingface.co/%s)" % HF_REPO_ID)

    # INPUTS
    st.markdown("## Employee Inputs")

    # Initialize session state for synced fields
    if "satisfaction_level" not in st.session_state:
        st.session_state.satisfaction_level = 0.50
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = 0.70

    def sync_sat_slider():
        st.session_state.satisfaction_level = st.session_state.sat_slider

    def sync_sat_input():
        st.session_state.satisfaction_level = st.session_state.sat_input

    def sync_eval_slider():
        st.session_state.last_evaluation = st.session_state.eval_slider

    def sync_eval_input():
        st.session_state.last_evaluation = st.session_state.eval_input

    # Use a form for better UX + validation + fewer reruns
    with st.form("turnover_form", clear_on_submit=False, border=False):
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown('<div class="card riseIn">', unsafe_allow_html=True)
            st.markdown("<div class='cardTitle'>Satisfaction</div>", unsafe_allow_html=True)
            s1, s2 = st.columns([3, 1])
            with s1:
                st.slider(
                    "Satisfaction",
                    0.0, 1.0,
                    value=float(st.session_state.satisfaction_level),
                    step=0.01,
                    key="sat_slider",
                    on_change=sync_sat_slider,
                    label_visibility="collapsed",
                    help="0 = Very dissatisfied, 1 = Very satisfied",
                )
            with s2:
                st.number_input(
                    "Satisfaction value",
                    0.0, 1.0,
                    value=float(st.session_state.satisfaction_level),
                    step=0.01,
                    format="%.2f",
                    key="sat_input",
                    on_change=sync_sat_input,
                    label_visibility="collapsed",
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card riseIn">', unsafe_allow_html=True)
            st.markdown("<div class='cardTitle'>Last Evaluation</div>", unsafe_allow_html=True)
            e1, e2 = st.columns([3, 1])
            with e1:
                st.slider(
                    "Evaluation",
                    0.0, 1.0,
                    value=float(st.session_state.last_evaluation),
                    step=0.01,
                    key="eval_slider",
                    on_change=sync_eval_slider,
                    label_visibility="collapsed",
                    help="0 = Poor, 1 = Excellent",
                )
            with e2:
                st.number_input(
                    "Evaluation value",
                    0.0, 1.0,
                    value=float(st.session_state.last_evaluation),
                    step=0.01,
                    format="%.2f",
                    key="eval_input",
                    on_change=sync_eval_input,
                    label_visibility="collapsed",
                )
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4, c5 = st.columns(3, gap="large")
        with c3:
            time_spend_company = st.number_input(
                "Years at Company",
                min_value=1, max_value=40, value=3, step=1,
                help="Tenure in years",
            )
        with c4:
            number_project = st.number_input(
                "Number of Projects",
                min_value=1, max_value=10, value=4, step=1,
                help="Concurrent projects",
            )
        with c5:
            average_monthly_hours = st.number_input(
                "Average Monthly Hours",
                min_value=80, max_value=350, value=200, step=5,
                help="Average working hours per month",
            )

        # lightweight validation feedback
        if average_monthly_hours >= 260:
            st.warning("Very high monthly hours may increase turnover risk in many workplaces.")

        st.markdown('<div class="hr-grad"></div>', unsafe_allow_html=True)

        b1, b2, b3 = st.columns([1, 2, 1])
        with b2:
            submitted = st.form_submit_button("Predict Turnover", use_container_width=True)

        # Optional reset (doesn't clear form automatically; resets key session values)
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            reset = st.form_submit_button("Reset Satisfaction / Evaluation", use_container_width=True)

    if reset:
        st.session_state.satisfaction_level = 0.50
        st.session_state.last_evaluation = 0.70
        st.toast("Reset complete.", icon="✅")

    if not submitted:
        st.markdown(
            "<div class='small-muted'>Enter values above and click Predict to see the results.</div>",
            unsafe_allow_html=True,
        )
        return

    # Build input
    input_data = {
        "satisfaction_level": float(st.session_state.satisfaction_level),
        "time_spend_company": int(time_spend_company),
        "average_monthly_hours": int(average_monthly_hours),
        "number_project": int(number_project),
        "last_evaluation": float(st.session_state.last_evaluation),
    }

    input_df = pd.DataFrame([input_data])[BEST_FEATURES]

    with st.spinner("Running prediction..."):
        prediction = int(model.predict(input_df)[0])
        proba = model.predict_proba(input_df)[0]
        prob_stay = float(proba[0]) * 100.0
        prob_leave = float(proba[1]) * 100.0

    st.toast("Prediction generated.", icon="🔮")

    st.markdown('<div class="hr-grad"></div>', unsafe_allow_html=True)
    st.markdown("## Results")

    left, right = st.columns([1.15, 0.85], gap="large")

    # RESULT CARD
    with left:
        if prediction == 0:
            badge_class = "good"
            headline = "Likely to STAY"
            detail = f"Model estimates a low-to-moderate leave risk ({prob_leave:.1f}%)."
        else:
            badge_class = "bad"
            headline = "Likely to LEAVE"
            detail = f"Model estimates an elevated leave risk ({prob_leave:.1f}%)."

        st.markdown(
            f"""
<div class="card riseIn">
  <div class="badge {badge_class}">Prediction</div>
  <div style="height:10px"></div>
  <div class="cardKPI">{headline}</div>
  <div class="small-muted" style="margin-top:8px;">{detail}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        # Recommendations
        recs = recommendations(input_data, prob_leave / 100.0)
        st.markdown(
            """
<div class="card riseIn" style="margin-top:14px;">
  <div class="cardTitle">Recommended Next Steps</div>
  <div class="small-muted">These are guidance-oriented suggestions, not causal explanations.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        for r in recs:
            st.write(f"- {r}")

    # PROBABILITIES VISUAL
    with right:
        st.markdown(
            f"""
<div class="card riseIn">
  <div class="cardTitle">Probability Overview</div>
  <div class="small-muted">Stay vs Leave</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        if PLOTLY_OK:
            fig = make_gauge(prob_leave_pct=prob_leave)
            # Adjust gauge text for light mode
            if st.session_state.theme_mode == "light":
                fig.update_layout(font=dict(color="#0B1220"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            # Fallback: sleek bars with the same palette logic
            st.write(f"Stay: {prob_stay:.1f}%")
            st.progress(min(max(prob_stay / 100.0, 0.0), 1.0))
            st.write(f"Leave: {prob_leave:.1f}%")
            st.progress(min(max(prob_leave / 100.0, 0.0), 1.0))
            st.caption("Install plotly for a gauge visualization: `pip install plotly`")

        # Export block
        export_payload = {
            "input": input_data,
            "prediction_class": prediction,
            "probability_stay_pct": round(prob_stay, 3),
            "probability_leave_pct": round(prob_leave, 3),
            "features_order": BEST_FEATURES,
        }
        st.download_button(
            "Download Result (JSON)",
            data=json.dumps(export_payload, indent=2),
            file_name="turnover_prediction.json",
            mime="application/json",
            use_container_width=True,
        )

    # INPUT SUMMARY + FEATURE IMPORTANCE
    st.markdown('<div class="hr-grad"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        with st.expander("Input Summary", expanded=False):
            summary_df = pd.DataFrame(
                {
                    "Feature": [
                        "Satisfaction Level",
                        "Last Evaluation",
                        "Years at Company",
                        "Number of Projects",
                        "Average Monthly Hours",
                    ],
                    "Value": [
                        f"{input_data['satisfaction_level']:.2f}",
                        f"{input_data['last_evaluation']:.2f}",
                        f"{input_data['time_spend_company']}",
                        f"{input_data['number_project']}",
                        f"{input_data['average_monthly_hours']}",
                    ],
                }
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with c2:
        with st.expander("Model Signals (Feature Importance)", expanded=False):
            # RandomForestClassifier exposes feature_importances_ when trained with sklearn
            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame(
                    {
                        "feature": BEST_FEATURES,
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                st.bar_chart(fi.set_index("feature"))
                st.caption("Importance values come from the trained Random Forest model.")
            else:
                st.info("This model object does not expose `feature_importances_`.")

    st.markdown(
        """
<div class="small-muted" style="margin-top:18px;">
  Built with Streamlit. Model loaded from Hugging Face Hub.
</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
