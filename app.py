import os
import re
from html import escape
from sklearn.utils.validation import check_is_fitted

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# ✅ ensure tfidf is trained (CRITICAL FOR DEPLOYMENT)
try:
    check_is_fitted(tfidf)
except:
    st.error("❌ TF-IDF is not fitted. Please upload correct trained tfidf.pkl")
    st.stop()

EXAMPLES = {
    "Choose an example": "",
    "Casual check-in": "Hey, are you free to talk for a few minutes today?",
    "Urgent follow-up": "Please reply asap, I need your confirmation urgently.",
    "Relationship tension": "I feel like you never listen when I try to explain this.",
    "Simple reminder": "Just reminding you about the message I sent yesterday.",
}

CLASS_HINTS = {
    "Busy": "Likely delayed by time or attention pressure.",
    "Forgot": "Likely low urgency or slipped attention.",
    "Ignoring": "May be avoiding the conversation.",
    "Not Interested": "May indicate weak interest in continuing.",
}

CLASS_COLORS = {
    "Busy": "#0f766e",
    "Forgot": "#2563eb",
    "Ignoring": "#b45309",
    "Not Interested": "#be123c",
}


def clean_message(message):
    return re.sub(r"\s+", " ", message.strip())


def get_delay_label(minutes):
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f} hr"
    return f"{hours / 24:.1f} days"


def get_confidence_tone(confidence):
    if confidence >= 0.75:
        return "Strong signal"
    if confidence >= 0.55:
        return "Moderate signal"
    return "Mixed signal"


def get_reliability_note(ranked):
    top_label, top_probability = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else ("", 0)
    margin = top_probability - runner_up[1]

    if top_probability < 0.45:
        return "The classes are close together, so treat this as uncertain."
    if margin < 0.12:
        return f"{top_label} is only slightly ahead of {runner_up[0]}."
    return "The leading class is meaningfully ahead of the alternatives."


def build_features(message, delay_minutes):
    try:
        text_vec = tfidf.transform([clean_message(message)])
    except:
        st.error("❌ TF-IDF transform failed. Model not trained properly.")
        st.stop()

    delay_norm = min(max(delay_minutes, 0), 2880) / 2880
    return hstack([text_vec, np.array([[delay_norm]])])


def predict_intent(message, delay_minutes):
    features = build_features(message, delay_minutes)
    expected_features = getattr(model, "n_features_in_", features.shape[1])
    if features.shape[1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, but app created {features.shape[1]}."
        )

    pred = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_ids = getattr(model, "classes_", np.arange(len(probabilities)))
    labels = label_encoder.inverse_transform(class_ids)
    order = np.argsort(probabilities)[::-1]

    return {
        "prediction": label_encoder.inverse_transform([pred])[0],
        "confidence": float(np.max(probabilities)),
        "labels": labels,
        "probabilities": probabilities,
        "ranked": [(str(labels[index]), float(probabilities[index])) for index in order],
    }


def render_probability_rows(labels, probabilities):
    rows = []
    for label, probability in sorted(
        zip(labels, probabilities), key=lambda item: item[1], reverse=True
    ):
        percent = float(probability) * 100
        color = CLASS_COLORS.get(label, "#0f766e")
        rows.append(
            f"""
            <div class="prob-row">
                <div class="prob-head">
                    <span>{escape(label)}</span>
                    <strong>{percent:.0f}%</strong>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {percent:.1f}%; background: {color}"></div>
                </div>
            </div>
            """
        )
    return "".join(rows)


def make_probability_chart(labels, probabilities):
    chart_data = pd.DataFrame(
        {
            "Intent": labels,
            "Probability": [
                round(float(probability) * 100, 1) for probability in probabilities
            ],
        }
    ).sort_values("Probability", ascending=False)
    return chart_data.set_index("Intent")


st.set_page_config(
    page_title="Intent Behind Silence Predictor",
    page_icon="chat",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                linear-gradient(135deg, rgba(210, 241, 235, 0.96) 0%, rgba(236, 228, 250, 0.94) 48%, rgba(255, 232, 206, 0.94) 100%),
                linear-gradient(180deg, #d2f1eb 0%, #ffe8ce 100%);
            color: #14231f;
        }

        .main .block-container {
            max-width: 980px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.2rem 0 0.9rem;
        }

        .eyebrow {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }

        .hero h1 {
            color: #14231f;
            font-size: clamp(2rem, 4.8vw, 3.15rem);
            line-height: 1.05;
            margin: 0;
        }

        .hero p {
            color: #344d47;
            font-size: 1.03rem;
            line-height: 1.7;
            max-width: 700px;
            margin-top: 0.85rem;
        }

        .status-grid, .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.25rem;
        }

        .insight-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin-top: 1rem;
        }

        .mini-card, .panel, .empty-state, .prob-panel, .insight-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(15, 118, 110, 0.16);
            border-radius: 8px;
            box-shadow: 0 18px 44px rgba(45, 63, 58, 0.13);
        }

        .mini-card, .insight-card {
            padding: 0.95rem;
        }

        .mini-card span, .insight-card span {
            color: #536a63;
            display: block;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .mini-card strong, .insight-card strong {
            color: #152823;
            display: block;
            font-size: 1.08rem;
            margin-top: 0.25rem;
        }

        .insight-card p {
            color: #415a53;
            font-size: 0.9rem;
            margin: 0.35rem 0 0;
        }

        .panel {
            padding: 1.25rem;
            margin-top: 1rem;
        }

        .panel-title {
            color: #152823;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0 0 0.2rem;
        }

        .panel-subtitle {
            color: #536a63;
            font-size: 0.92rem;
            margin: 0 0 1rem;
        }

        .result-card {
            background: linear-gradient(135deg, #092f2b 0%, #0e8074 58%, #d97706 100%);
            box-shadow: 0 20px 48px rgba(7, 39, 36, 0.26);
            border-radius: 8px;
            color: white;
            margin-top: 1.25rem;
            overflow: hidden;
            padding: 1.25rem;
            position: relative;
        }

        .result-card::after {
            background: rgba(255, 255, 255, 0.13);
            border-radius: 999px;
            content: "";
            height: 9rem;
            position: absolute;
            right: -3.5rem;
            top: -3.5rem;
            width: 9rem;
        }

        .result-label {
            color: #ccfbf1;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin: 0 0 0.25rem;
            text-transform: uppercase;
        }

        .result-value {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0;
            position: relative;
        }

        .confidence, .result-meta {
            color: #e4fffa;
            margin-top: 0.55rem;
            position: relative;
        }

        .result-meta {
            color: #d7fff7;
            font-size: 0.92rem;
            max-width: 38rem;
        }

        .prob-panel {
            margin-top: 1rem;
            padding: 1rem;
        }

        .prob-row {
            margin-top: 0.85rem;
        }

        .prob-head {
            align-items: center;
            color: #243b35;
            display: flex;
            font-size: 0.95rem;
            justify-content: space-between;
            margin-bottom: 0.35rem;
        }

        .prob-track {
            background: #d4e5df;
            border-radius: 999px;
            height: 0.72rem;
            overflow: hidden;
        }

        .prob-fill {
            border-radius: 999px;
            height: 100%;
        }

        .empty-state {
            margin-top: 1.1rem;
            padding: 1rem;
        }

        .empty-state strong {
            color: #152823;
            display: block;
            margin-bottom: 0.25rem;
        }

        .empty-state p {
            color: #536a63;
            margin: 0;
        }

        .footer-note {
            color: #405852;
            font-size: 0.88rem;
            margin-top: 1.25rem;
            text-align: center;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 118, 110, 0.16);
            border-radius: 8px;
            padding: 1rem;
        }

        .stButton > button {
            background: linear-gradient(135deg, #f97316 0%, #dc2626 100%);
            border: 0;
            border-radius: 8px;
            box-shadow: 0 14px 34px rgba(249, 115, 22, 0.32);
            color: white;
            font-weight: 800;
            padding: 0.8rem 1rem;
            width: 100%;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #ea580c 0%, #b91c1c 100%);
            color: white;
        }

        div[data-testid="stTabs"] button {
            font-weight: 700;
        }

        textarea, input {
            background-color: #f8fbfa !important;
            border-radius: 8px !important;
            color: #152823 !important;
        }

        label, div[data-testid="stMarkdownContainer"] p {
            color: #223a34;
        }

        div[data-baseweb="select"] > div {
            background-color: #f8fbfa;
            border-color: rgba(15, 118, 110, 0.24);
            color: #152823;
        }

        div[data-baseweb="select"] span {
            color: #152823;
        }

        div[data-testid="stTabs"] button p,
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricValue"],
        div[data-testid="stNumberInput"] label,
        div[data-testid="stTextArea"] label,
        div[data-testid="stSelectbox"] label {
            color: #14231f !important;
        }

        div[data-testid="stAlert"] {
            color: #14231f;
        }

        .stMarkdown, .stText {
            color: #14231f;
        }

        .panel,
        .empty-state,
        .prob-panel,
        .insight-card,
        .mini-card {
            color: #14231f;
        }

        .panel p,
        .empty-state p,
        .prob-panel p,
        .insight-card p,
        .mini-card p {
            color: #344d47 !important;
        }

        .panel-title,
        .empty-state strong,
        .prob-panel .panel-title,
        .insight-card strong,
        .mini-card strong,
        .prob-head,
        .prob-head span,
        .prob-head strong {
            color: #14231f !important;
        }

        .result-card,
        .result-card p,
        .result-card div {
            color: #e4fffa !important;
        }

        .result-card .result-label {
            color: #ccfbf1 !important;
        }

        .result-card .result-value {
            color: #ffffff !important;
        }

        textarea::placeholder,
        input::placeholder {
            color: #6b7f78 !important;
            opacity: 1 !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] input {
            color: #14231f !important;
            -webkit-text-fill-color: #14231f !important;
        }

        @media (max-width: 720px) {
            .status-grid, .insight-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">NLP prediction workspace</div>
        <h1>Intent Behind Silence Predictor</h1>
        <p>
            Analyze a message with its reply delay and get a clearer read on
            the likely intent behind the silence.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="status-grid">
        <div class="mini-card"><span>Model</span><strong>{escape(type(model).__name__)}</strong></div>
        <div class="mini-card"><span>Signals</span><strong>Message + delay</strong></div>
        <div class="mini-card"><span>Outputs</span><strong>{len(label_encoder.classes_)} intent classes</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

predict_tab, guide_tab = st.tabs(["Predict", "Guide"])

with predict_tab:
    st.markdown(
        """
        <div class="panel">
            <p class="panel-title">Message details</p>
            <p class="panel-subtitle">Write your message or load a quick sample.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_example = st.selectbox("Try a sample", list(EXAMPLES.keys()))
    message_default = EXAMPLES[selected_example] if selected_example != "Choose an example" else ""

    message = st.text_area(
        "Message",
        value=message_default,
        placeholder="Example: Hey, are you available to talk today?",
        height=140,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        delay = st.number_input("Delay", min_value=1, max_value=10080, value=60, step=5)
    with col2:
        delay_unit = st.selectbox("Unit", ["minutes", "hours"])
    with col3:
        message_words = len(clean_message(message).split())
        st.metric("Words", message_words)

    delay_minutes = delay * 60 if delay_unit == "hours" else delay

    st.markdown(
        f"""
        <div class="empty-state">
            <strong>Current input</strong>
            <p>{message_words} words measured with a {escape(get_delay_label(delay_minutes))} delay.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Predict"):
        cleaned_message = clean_message(message)
        if not cleaned_message:
            st.warning("Enter a message")
        else:
            result = predict_intent(cleaned_message, delay_minutes)
            prediction = result["prediction"]
            confidence = result["confidence"]
            labels = result["labels"]
            probabilities = result["probabilities"]
            ranked = result["ranked"]
            runner_up = ranked[1] if len(ranked) > 1 else ("", 0)

            st.markdown(
                f"""
                <div class="result-card">
                    <p class="result-label">Prediction</p>
                    <p class="result-value">{escape(prediction)}</p>
                    <div class="confidence">
                        {confidence:.0%} confidence | {escape(get_confidence_tone(confidence))}
                    </div>
                    <div class="result-meta">
                        {escape(CLASS_HINTS.get(prediction, "Review the full probability spread before acting."))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence < 0.55:
                st.warning("Low-confidence prediction. Use the probability spread, not only the top label.")

            st.markdown(
                f"""
                <div class="insight-grid">
                    <div class="insight-card">
                        <span>Runner-up</span>
                        <strong>{escape(runner_up[0]) if runner_up[0] else "N/A"} ({runner_up[1]:.0%})</strong>
                        <p>{escape(get_reliability_note(ranked))}</p>
                    </div>
                    <div class="insight-card">
                        <span>Delay signal</span>
                        <strong>{escape(get_delay_label(delay_minutes))}</strong>
                        <p>Very long delays are capped for model stability during scoring.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="prob-panel">
                    <p class="panel-title">Probability ranking</p>
                    {render_probability_rows(labels, probabilities)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="prob-panel"><p class="panel-title">Graph view</p></div>',
                unsafe_allow_html=True,
            )
            st.bar_chart(make_probability_chart(labels, probabilities), horizontal=True)
    else:
        st.markdown(
            """
            <div class="empty-state">
                <strong>Ready for prediction</strong>
                <p>Your result panel and probability breakdown will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with guide_tab:
    st.markdown(
        """
        <div class="panel">
            <p class="panel-title">How to read the result</p>
            <p class="panel-subtitle">
                The prediction is a model estimate, not a certainty. Use the confidence
                score and class probabilities together.
            </p>
        </div>
        <div class="status-grid">
            <div class="mini-card"><span>Busy</span><strong>May reply later</strong></div>
            <div class="mini-card"><span>Forgot</span><strong>Low urgency</strong></div>
            <div class="mini-card"><span>Ignoring</span><strong>Avoiding response</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer-note">Built for quick NLP inference with your saved model artifacts.</div>',
    unsafe_allow_html=True,
)
