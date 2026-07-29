# all the libraries my app needs
import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")               # render charts without a screen (needed on the server)
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# i import my own preprocessing so the app cleans the input the SAME way i trained
from features import preprocess

# page title, tab icon, and a wide layout so the HUD has room to breathe
st.set_page_config(page_title="IntentRadar", layout="wide")


# this whole block is my custom styling to give it the dark JARVIS/HUD look.
# i inject CSS with st.markdown because streamlit doesn't do this theming on its own.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&display=swap');
:root { --cyan:#22d3ee; --amber:#ffb020; --panel:#0e1622; --line:#1f3346;
        --head:'Gowun Batang',serif; --body:'Gowun Batang',serif; }

/* force my font on everything, including the dropdown popup (it renders outside the app,
   so i list the popover/menu bits too, otherwise the options stay the default font) */
html, body, .stApp, .stApp p, .stMarkdown, label, input, textarea, button,
[data-baseweb="select"] *, [data-baseweb="popover"] *, [data-baseweb="menu"] *,
[role="option"], [role="listbox"] * { font-family:var(--body) !important; }

/* dark background with a faint glow and scan-line grid for the HUD feel */
.stApp {
    background:
        radial-gradient(circle at 20% 10%, rgba(34,211,238,.06), transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(34,211,238,.05), transparent 40%),
        repeating-linear-gradient(0deg, transparent, transparent 38px, rgba(34,211,238,.03) 39px),
        #0a0e14;
    color:#d7e6ee;
}
h1,h2,h3,h4 { font-family:var(--head) !important; letter-spacing:1px; }

/* my font sizes, bumped up so everything is easy to read */
.stApp, .stApp p, .stMarkdown { font-size:1.08rem; font-weight:500; }
label, [data-testid="stWidgetLabel"] p { font-size:1.08rem !important; font-weight:500; }
h1 { font-size:2.5rem !important; }
h2 { font-size:1.4rem !important; }

/* the numbers i type, dropdown text and radios, made bigger too */
input, textarea, .stNumberInput input { font-size:1.05rem !important; }
[data-baseweb="select"] * { font-size:1.05rem !important; }
.stRadio label p, [data-testid="stExpander"] summary,
[data-testid="stExpander"] p { font-size:1.05rem !important; }
.stButton>button { font-size:1.1rem !important; padding:.6rem 1rem !important; }

/* my two tabs (scanner / about) styled to match the HUD */
[data-baseweb="tab-list"] { gap:1.5rem; border-bottom:1px solid var(--line); }
[data-baseweb="tab"] { font-family:var(--body) !important; letter-spacing:2px;
                       text-transform:uppercase; color:#6fb8cc; }
[data-baseweb="tab"][aria-selected="true"] { color:#8ef1ff !important; }

/* a reusable HUD panel: dark box, cyan border, little corner brackets */
.hud {
    position:relative; background:var(--panel);
    border:1px solid var(--line); border-radius:4px;
    padding:1rem 1.2rem; margin:.4rem 0;
    box-shadow:0 0 18px rgba(34,211,238,.07) inset;
}
.hud::before, .hud::after {
    content:""; position:absolute; width:14px; height:14px; border:2px solid var(--cyan);
}
.hud::before { top:-1px; left:-1px; border-right:0; border-bottom:0; }
.hud::after  { bottom:-1px; right:-1px; border-left:0; border-top:0; }

/* small uppercase cyan labels i use as section headers */
.label { font-family:var(--body); color:#6fb8cc; font-size:.88rem; font-weight:600;
         letter-spacing:2px; text-transform:uppercase; }

/* the verdict box, glows cyan for buy and amber for no-buy */
.verdict {
    font-family:var(--body); text-align:center;
    padding:1.1rem; border-radius:4px; letter-spacing:2px;
}
.verdict.buy   { border:1px solid var(--cyan);  color:#8ef1ff;
                 box-shadow:0 0 22px rgba(34,211,238,.35); background:rgba(34,211,238,.06); }
.verdict.nobuy { border:1px solid var(--amber); color:#ffd27a;
                 box-shadow:0 0 22px rgba(255,176,32,.25); background:rgba(255,176,32,.05); }
.verdict .big  { font-size:1.6rem; font-weight:700; margin:.2rem 0; }

/* my glowing cyan button */
.stButton>button {
    font-family:var(--body) !important; letter-spacing:2px; text-transform:uppercase;
    background:rgba(34,211,238,.08) !important; color:#8ef1ff !important;
    border:1px solid var(--cyan) !important; border-radius:3px !important;
}
.stButton>button:hover { box-shadow:0 0 18px rgba(34,211,238,.5) !important; background:rgba(34,211,238,.18) !important; }

/* sidebar styled like a system console */
[data-testid="stSidebar"] { background:#0b1119; border-right:1px solid var(--line); }
.sys-line { font-family:var(--body); font-size:.95rem; font-weight:500; color:#9fd6e4; margin:.2rem 0; }
.sys-key  { color:#5a7a88; }

/* about-page bullet lines */
.about-line { font-size:1.05rem; color:#cfe8f0; margin:.35rem 0; }
.about-line b { color:#8ef1ff; }
</style>
""", unsafe_allow_html=True)


# load my trained model. @st.cache_resource means it only loads once, not on every click
@st.cache_resource
def load_model():
    # path relative to this file so it works after i deploy too
    path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
    art = joblib.load(path)
    # i saved the model, its training columns AND the tuned decision threshold together.
    # the threshold matters: my notebook tunes it to ~0.3 (not the default 0.5) because the
    # data is imbalanced, so if the app ignored it the app would disagree with my notebook.
    # .get() with a 0.5 fallback keeps this working with an older pkl that has no threshold.
    return art["model"], list(art["columns"]), float(art.get("threshold", 0.5))

# if the model file is missing, show a clear message instead of crashing
try:
    model, train_columns, THRESHOLD = load_model()
except Exception as e:
    st.error(f"SYSTEM OFFLINE. Could not load models/model.pkl. Details: {e}")
    st.stop()


# a "typical" session (medians / most common values from my training data).
# my explainer swaps each signal to this to see how much that signal changed the result.
# i only need entries for the features EXPLAIN actually touches below
BASELINE = {
    "Administrative": 1.0, "ProductRelated": 18.0, "ProductRelated_Duration": 608.94,
    "BounceRates": 0.0029, "ExitRates": 0.025, "PageValues": 0.0,
    "Month": "May", "VisitorType": "Returning_Visitor",
}
# the signals my explainer reports on, with friendly names to show the user
EXPLAIN = {
    "PageValues": "Page value", "ProductRelated": "Product pages",
    "ProductRelated_Duration": "Time on products", "ExitRates": "Exit rate",
    "BounceRates": "Bounce rate", "Administrative": "Admin pages",
    "Month": "Month", "VisitorType": "Visitor type",
}


# helper: preprocess one session the same as training, line up the columns, return buy probability
def _proba(row_df):
    x = preprocess(row_df).reindex(columns=train_columns, fill_value=0)
    return float(model.predict_proba(x)[0, 1])


def explain(row_df):
    # my "why this prediction" logic. i take the real probability, then for each signal i
    # swap it to the typical value and predict again. the difference tells me how much that
    # signal pushed the result up (toward buy) or down (toward no-buy). uses only my model.
    base = _proba(row_df)
    effects = []
    for feat, label in EXPLAIN.items():
        mod = row_df.copy()
        mod.loc[:, feat] = BASELINE[feat]
        effects.append((label, base - _proba(mod)))   # positive means it pushed toward BUY
    effects.sort(key=lambda t: abs(t[1]), reverse=True)   # biggest effect first
    return effects[:6]


def gauge(proba):
    # my arc-reactor style dial. i draw a dark ring, then a cyan (or amber) glowing arc
    # that fills up to the probability, with the % in the middle.
    pct = proba * 100
    color = "#22d3ee" if proba >= THRESHOLD else "#ffb020"   # same cut-off as the verdict, so they agree
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    fig.patch.set_alpha(0.0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    ax.add_patch(Wedge((0, 0), 1.0, 0, 360, width=0.16, facecolor="#16202e"))   # background ring
    start, span = 90, proba * 360                                              # start at top, fill clockwise
    for w, a in [(0.34, 0.06), (0.24, 0.13)]:                                  # faint wider arcs make the glow
        ax.add_patch(Wedge((0, 0), 1.0, start - span, start, width=w, facecolor=color, alpha=a))
    ax.add_patch(Wedge((0, 0), 1.0, start - span, start, width=0.16, facecolor=color))
    ax.add_patch(Circle((0, 0), 0.62, facecolor="#0a0e14", edgecolor=color, lw=1.2, alpha=0.9))
    ax.text(0, 0.06, f"{pct:.0f}%", ha="center", va="center", color=color,
            fontsize=32, fontweight="bold", fontfamily="DejaVu Sans")
    ax.text(0, -0.30, "INTENT", ha="center", va="center", color="#7dd3e8",
            fontsize=11, fontfamily="DejaVu Sans")
    return fig


def drivers_chart(effects):
    # horizontal bars of what drove the call. cyan bars push toward buy, amber push toward no-buy.
    labels = [e[0] for e in effects][::-1]
    vals = [e[1] * 100 for e in effects][::-1]
    colors = ["#22d3ee" if v >= 0 else "#ffb020" for v in vals]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    ax.barh(range(len(labels)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, color="#cfe8f0", fontfamily="DejaVu Sans", fontsize=9)
    ax.axvline(0, color="#3a4a5a", lw=1)
    ax.tick_params(colors="#5a6b7a", labelsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel("effect on purchase probability  (%)", color="#7dd3e8", fontsize=8, fontfamily="DejaVu Sans")
    return fig


def play_video(kind):
    # play my higgsfield HUD clip fullscreen over everything after a scan (buy or nobuy).
    # it covers the whole screen (and the result underneath) while it plays, then fades out
    # and disappears when the clip ends, revealing the result. muted so autoplay is allowed.
    # streamlit serves the file from the static/ folder (enableStaticServing in config.toml).
    # for the buy clip i add #t=1 so it skips the slow first second and jumps to the cash burst.
    start = "#t=1" if kind == "buy" else ""
    play_secs = 4 if kind == "buy" else 5      # how long the clip actually plays
    src = f"app/static/{kind}.mp4{start}"
    st.markdown(f"""
    <div class="vid-overlay">
      <video autoplay muted playsinline>
        <source src="{src}" type="video/mp4">
      </video>
    </div>
    <style>
    .vid-overlay {{
        position:fixed; inset:0; z-index:999999; pointer-events:none;
        background:#0a0e14; overflow:hidden;
        animation:vidhide {play_secs}s forwards;
    }}
    .vid-overlay video {{ width:100%; height:100%; object-fit:cover; }}
    @keyframes vidhide {{ 0%,88% {{opacity:1;}} 100% {{opacity:0; visibility:hidden;}} }}
    </style>
    """, unsafe_allow_html=True)


# two ready-made scenarios so i can fill every input in one click during my demo.
# high-intent should come out as buy, casual browser as no-buy.
PRESETS = {
    "Custom (enter your own below)": None,
    "High-intent shopper": dict(
        Administrative=3, Administrative_Duration=80.0, Informational=1,
        Informational_Duration=20.0, ProductRelated=40, ProductRelated_Duration=1200.0,
        BounceRates=0.005, ExitRates=0.01, PageValues=35.0, SpecialDay=0.0,
        Month="Nov", VisitorType="Returning_Visitor", TrafficType=2,
        OperatingSystems=2, Browser=2, Region=1, Weekend=True),
    "Casual browser": dict(
        Administrative=0, Administrative_Duration=0.0, Informational=0,
        Informational_Duration=0.0, ProductRelated=2, ProductRelated_Duration=15.0,
        BounceRates=0.2, ExitRates=0.2, PageValues=0.0, SpecialDay=0.0,
        Month="Feb", VisitorType="New_Visitor", TrafficType=1,
        OperatingSystems=1, Browser=1, Region=3, Weekend=False),
}
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITORS = ["Returning_Visitor", "New_Visitor", "Other"]

# my running list of every scan this session (inputs + result), kept in session_state so it
# survives reruns. i show it as a table and let the user download it as a CSV.
if "history" not in st.session_state:
    st.session_state.history = []


def render_scanner():
    # pick a preset to auto-fill the inputs, or Custom to enter your own
    preset_name = st.selectbox("▸ Load scenario", list(PRESETS.keys()))
    P = PRESETS[preset_name] or PRESETS["High-intent shopper"]

    # my inputs, split into two columns with plain-english labels the target audience understands
    st.markdown('<span class="label">Session signals</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        product_related = st.number_input("Product pages viewed", 0, 800, int(P["ProductRelated"]))
        product_dur = st.number_input("Time on product pages (sec)", 0.0, 60000.0, float(P["ProductRelated_Duration"]))
        admin = st.number_input("Account/admin pages viewed", 0, 100, int(P["Administrative"]))
        admin_dur = st.number_input("Time on admin pages (sec)", 0.0, 6000.0, float(P["Administrative_Duration"]))
        info = st.number_input("Info pages viewed", 0, 100, int(P["Informational"]))
        info_dur = st.number_input("Time on info pages (sec)", 0.0, 6000.0, float(P["Informational_Duration"]))
    with c2:
        page_values = st.number_input("Page Value (Google Analytics)", 0.0, 400.0, float(P["PageValues"]),
                                      help="Average value of the pages visited this session. Strongest signal in the model.")
        bounce = st.slider("Bounce rate", 0.0, 1.0, float(P["BounceRates"]), 0.005)
        exit_rate = st.slider("Exit rate", 0.0, 1.0, float(P["ExitRates"]), 0.005)
        month = st.selectbox("Month", MONTHS, index=MONTHS.index(P["Month"]))
        visitor = st.selectbox("Visitor type", VISITORS, index=VISITORS.index(P["VisitorType"]))
        traffic = st.number_input("Traffic source code", 1, 20, int(P["TrafficType"]),
                                  help="Which channel brought the visitor in (1-20).")

    # a second row for the remaining raw features, my final model (cycle 1) uses all 17
    st.markdown('<span class="label">Other session details</span>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        special_day = st.slider("Closeness to a special day", 0.0, 1.0, float(P["SpecialDay"]), 0.1,
                                help="How close the session is to a special day (e.g. Valentine's, Mother's Day).")
        weekend = st.checkbox("Weekend session", bool(P["Weekend"]))
    with c4:
        os_code = st.number_input("Operating system code", 1, 8, int(P["OperatingSystems"]))
        browser_code = st.number_input("Browser code", 1, 13, int(P["Browser"]))
        region_code = st.number_input("Region code", 1, 9, int(P["Region"]))

    # the widgets already block out-of-range numbers, but i check weird combinations myself
    errors = []
    if (product_related + admin + info) == 0 and (product_dur + admin_dur + info_dur) > 0:
        errors.append("Time was recorded but no pages were viewed, please check the inputs.")
    if bounce > exit_rate + 1e-9:
        errors.append("Bounce rate cannot be higher than exit rate for a session.")

    st.write("")
    # everything below only runs when the scan button is clicked
    if st.button("▸ SCAN SESSION", type="primary", use_container_width=True):
        if errors:
            for msg in errors:
                st.error(msg)
        else:
            # try/except so a bad input shows a message instead of crashing the app
            try:
                row = dict(
                    Administrative=admin, Administrative_Duration=admin_dur,
                    Informational=info, Informational_Duration=info_dur,
                    ProductRelated=product_related, ProductRelated_Duration=product_dur,
                    BounceRates=bounce, ExitRates=exit_rate, PageValues=page_values,
                    SpecialDay=special_day, Weekend=weekend,
                    OperatingSystems=os_code, Browser=browser_code, Region=region_code,
                    TrafficType=traffic, Month=month, VisitorType=visitor)
                row_df = pd.DataFrame([row])

                # a short pause + spinner to make it feel like it's really scanning
                with st.spinner("▸ ANALYZING SESSION SIGNALS..."):
                    time.sleep(0.7)
                    proba = _proba(row_df)
                    # compare against my TUNED threshold from the notebook, not 0.5
                    will_buy = proba >= THRESHOLD
                    effects = explain(row_df)
            except Exception as e:
                st.error(f"Scan failed, please check the inputs. Details: {e}")
                st.stop()

            # log this scan (the result first, then every input) to the history, newest on top
            st.session_state.history.insert(0, {
                "Decision": "Buy" if will_buy else "No buy",
                "Probability %": round(proba * 100, 1),
                **row,
            })

            # play the matching HUD clip over the whole screen, then it fades to reveal the result
            play_video("buy" if will_buy else "nobuy")

            # left: the gauge. right: the verdict card and recommended action.
            g_col, v_col = st.columns([1, 1.3])
            with g_col:
                st.pyplot(gauge(proba), use_container_width=True)
            with v_col:
                if will_buy:
                    st.markdown(
                        f'<div class="verdict buy"><div class="label">target acquired</div>'
                        f'<div class="big">◉ HIGH INTENT</div>{proba*100:.1f}% probability of purchase</div>',
                        unsafe_allow_html=True)
                    st.markdown('<div class="hud"><span class="label">Recommended action</span><br>'
                                'Trigger a real-time nudge: a free-shipping banner, live-chat offer, '
                                'or a small discount.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="verdict nobuy"><div class="label">low signal</div>'
                        f'<div class="big">○ LOW INTENT</div>{proba*100:.1f}% probability of purchase</div>',
                        unsafe_allow_html=True)
                    st.markdown('<div class="hud"><span class="label">Recommended action</span><br>'
                                'Hold incentives and let the visitor browse, to avoid wasting discounts.</div>',
                                unsafe_allow_html=True)

            # the "why this prediction" section: the bar chart on the left, readable bullets on the right
            st.write("")
            st.markdown('<span class="label">Signal analysis: what is driving this call</span>', unsafe_allow_html=True)
            d_col, r_col = st.columns([1.3, 1])
            with d_col:
                st.pyplot(drivers_chart(effects), use_container_width=True)
            with r_col:
                for label, delta in effects:
                    arrow = "▲ toward BUY" if delta >= 0 else "▼ toward NO-BUY"
                    col = "#22d3ee" if delta >= 0 else "#ffb020"
                    st.markdown(
                        f'<div class="sys-line"><span style="color:{col}">{arrow}</span> {label}</div>',
                        unsafe_allow_html=True)

    # the prediction history table. it shows whenever i have at least one scan this session.
    # st.dataframe with a fixed height shows about 10 rows and scrolls for the rest.
    if st.session_state.history:
        st.write("")
        st.markdown('<span class="label">Prediction history (scroll for older scans)</span>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, height=360, use_container_width=True)

        # download the whole history (every input + its prediction) as one CSV file
        csv = hist_df.to_csv(index=False).encode("utf-8")
        dl_col, clr_col = st.columns([1, 1])
        dl_col.download_button("⬇ Download history (CSV)", csv,
                               "intentradar_predictions.csv", "text/csv",
                               use_container_width=True)
        if clr_col.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()


def render_about():
    st.markdown('<span class="label">What it does</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud"><div class="about-line">Most online store visitors never buy. Sending an '
        'incentive to everyone wastes margin; sending none loses winnable sales. <b>IntentRadar</b> reads '
        'a live session and predicts whether it will end in a purchase, so the team can nudge only the '
        'high-intent shoppers, at the right moment.</div></div>', unsafe_allow_html=True)

    st.markdown('<span class="label">The data</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud"><div class="about-line"><b>UCI Online Shoppers Purchasing Intention</b> dataset: '
        '12,330 real browsing sessions, 17 features, and a Revenue flag (did the session end in a purchase). '
        'Only about 15% of sessions convert, which makes this an imbalanced problem.</div></div>',
        unsafe_allow_html=True)

    st.markdown('<span class="label">The model</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud">'
        '<div class="about-line"><b>Gradient Boosting</b> classifier (scikit-learn). All four candidate '
        'models were tuned with RandomizedSearchCV <i>before</i> being compared, so the winner was picked '
        'at its own best setting rather than at whatever its defaults happened to give.</div>'
        '<div class="about-line">F1 <b>0.684</b> &nbsp;·&nbsp; ROC-AUC <b>0.94</b> &nbsp;·&nbsp; recall <b>0.73</b>. '
        'I judge on F1 (not accuracy) because a do-nothing model is already ~85% accurate but catches zero '
        'buyers.</div></div>', unsafe_allow_html=True)

    st.markdown('<span class="label">Iterative development</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud">'
        '<div class="about-line"><b>Cycle 1 (deployed)</b>: 5 models trained on the full 17 raw '
        'features (all one-hot encoded), with the 4 real algorithms each tuned via RandomizedSearchCV '
        '(2 hyperparameters, 5-fold CV, scored on F1). Gradient Boosting won with F1 <b>0.68</b>.</div>'
        '<div class="about-line"><b>Cycle 2 (tested, rejected)</b>: engineered 4 combined features '
        '(total pages, total duration, etc), then retuned Gradient Boosting the same way. F1 dropped, '
        'the model can already learn these interactions on its own, so i kept the raw features.</div>'
        '<div class="about-line"><b>Cycle 3 (tested, rejected)</b>: used Cycle 1\'s feature importance '
        'to drop the 5 weakest features, then retuned again. F1 improved over the untuned version but '
        'still fell short of Cycle 1, so the full feature set stayed in.</div>'
        '<div class="about-line">Both later cycles were fair, tuned attempts and both honestly did not '
        'beat the properly-tuned baseline, so <b>Cycle 1 is the model deployed here</b>, using all 17 '
        'raw features, which is why every session detail below is a live, meaningful input.</div>'
        '<div class="about-line"><b>Decision threshold (accepted)</b>: the one change that did work. '
        'Only ~15% of sessions convert, so the default 0.5 cut-off is too strict. I picked the threshold '
        'on out-of-fold training predictions (never the test set) and it landed on <b>0.31</b>, raising '
        'F1 and lifting recall from 0.62 to <b>0.73</b>, the model now catches far more real buyers. It beat '
        '0.5 on 8 out of 8 different train/test splits, so it is a real effect, not luck. This app uses '
        'that exact threshold, loaded from the saved model file.</div>'
        '</div>', unsafe_allow_html=True)

    st.markdown('<span class="label">How the "why" works</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud"><div class="about-line">For each scan, the signal analysis takes one feature at a '
        'time, swaps it to a typical value, and re-runs the model. The change in probability shows how much '
        'that signal pushed the prediction toward BUY or NO-BUY. It uses only the trained model, no extra '
        'libraries.</div></div>', unsafe_allow_html=True)

    st.markdown('<span class="label">Built with</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud"><div class="about-line">scikit-learn · pandas · numpy · matplotlib · Streamlit. '
        'The app shares the exact same preprocessing as training (features.py), so there is no '
        'train/serve skew.</div></div>', unsafe_allow_html=True)


# the sidebar, styled like a system console with my branding, model stats and a how-it-works
with st.sidebar:
    st.markdown("## INTENTRADAR")
    st.markdown('<span class="label">real-time purchase-intent detection</span>', unsafe_allow_html=True)
    st.markdown('<div class="sys-line">◉ SYSTEM ONLINE</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<span class="label">Model readout</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sys-line"><span class="sys-key">engine  </span> Gradient Boosting (tuned)</div>'
        '<div class="sys-line"><span class="sys-key">F1      </span> 0.684</div>'
        '<div class="sys-line"><span class="sys-key">ROC-AUC </span> 0.94</div>'
        '<div class="sys-line"><span class="sys-key">recall  </span> 0.73</div>'
        f'<div class="sys-line"><span class="sys-key">cut-off </span> {THRESHOLD:.2f} (tuned)</div>'
        '<div class="sys-line"><span class="sys-key">dataset </span> UCI Online Shoppers</div>',
        unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<span class="label">How it works</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sys-line">1. feed the live session signals</div>'
        '<div class="sys-line">2. scan the session and read its intent</div>'
        '<div class="sys-line">3. see the drivers and the call</div>',
        unsafe_allow_html=True)


# the page header
st.markdown("# INTENTRADAR")
st.markdown('<span class="label">Scanning a live session for purchase intent, so the team '
            'can nudge high-intent shoppers at the right moment</span>', unsafe_allow_html=True)
st.write("")


# two tabs: the scanner (my app) and an about page describing the model
tab_scan, tab_about = st.tabs(["◉  SCANNER", "ℹ  ABOUT"])
with tab_scan:
    render_scanner()
with tab_about:
    render_about()


# a small footer line
st.write("")
st.markdown('<span class="label">Tuned scikit-learn classifier. UCI Online Shoppers dataset. '
            'Same preprocessing as training.</span>', unsafe_allow_html=True)
