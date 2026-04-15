import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(
    page_title="TIG Moderation AI",
    page_icon="🧠",
    layout="wide"
)

API = "http://127.0.0.1:8000/v1/predict"
BATCH = "http://127.0.0.1:8000/v1/predict_batch"

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<style>
.main-title {
    text-align:center;
    font-size:42px;
    font-weight:800;
    background: linear-gradient(90deg, #4F8BF9, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    padding:20px;
    border-radius:15px;
    background:#111827;
    box-shadow:0px 4px 20px rgba(0,0,0,0.3);
}

.result-safe {
    color:#22c55e;
    font-size:24px;
    font-weight:700;
}

.result-risk {
    color:#ef4444;
    font-size:24px;
    font-weight:700;
}

.result-uncertain {
    color:#f59e0b;
    font-size:24px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🧠 Tigrinya Harmful Content AI</div>", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([2,1])

with col1:
    text = st.text_area("Enter text", height=150, placeholder="Type Tigrinya or English text...")

with col2:
    st.metric("Characters", len(text))
    st.metric("Words", len(text.split()) if text else 0)

if st.button("🚀 Analyze"):

    if len(text.strip()) < 3:
        st.warning("Minimum 3 characters required")
    else:
        start = time.time()

        r = requests.post(API, json={"content": text})
        latency = time.time() - start

        if r.status_code == 200:

            d = r.json()
            p = d["confidence"]

            if p >= 0.65:
                label = "🔴 Harmful"
                style = "result-risk"
            elif p >= 0.40:
                label = "🟡 Uncertain"
                style = "result-uncertain"
            else:
                label = "🟢 Neutral"
                style = "result-safe"

            st.markdown(f"<div class='{style}'>{label}</div>", unsafe_allow_html=True)

            st.progress(float(p))

            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{p:.2f}")
            c2.metric("Latency", f"{latency:.3f}s")
            c3.metric("Model Label", d["label"])

            st.info(d["explanation"])

            st.session_state.history.append({
                "text": text,
                "label": label,
                "confidence": p,
                "latency": latency
            })

st.markdown("---")

st.subheader("📊 History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df.tail(10))