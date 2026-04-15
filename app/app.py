import streamlit as st
import requests
import pandas as pd
import time

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Harmful Content Moderation | Pro",
    page_icon="🧠",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

# -----------------------------
# SIDEBAR (PRO PANEL)
# -----------------------------
with st.sidebar:
    st.title("🧠 Moderation AI Pro")

    mode = st.radio(
        "Choose Mode",
        ["Single Prediction", "Batch CSV"]
    )

    st.markdown("---")
    st.info("XGBoost + Feature Engineering + FastAPI")
    st.markdown("⚡ Real-time moderation system")

# -----------------------------
# HEADER
# -----------------------------
st.title("🚨 Content Moderation Dashboard (Pro)")
st.caption("Detect Harmful vs Neutral content with explainable AI")

# =============================
# SINGLE PREDICTION MODE
# =============================
if mode == "Single Prediction":

    col1, col2 = st.columns([2, 1])

    with col1:
        text = st.text_area(
            "Enter text to analyze",
            height=180,
            placeholder="Paste social media post, comment, message..."
        )

    with col2:
        st.metric("Characters", len(text))
        st.metric("Words", len(text.split()) if text else 0)

    if st.button("🔍 Analyze", use_container_width=True):

        if not text.strip():
            st.warning("Please enter text")
        else:
            start = time.time()

            try:
                response = requests.post(API_URL, json={"content": text})
                latency = time.time() - start

                if response.status_code == 200:
                    result = response.json()

                    label = result["label"]
                    confidence = float(result["confidence"])
                    explanation = result.get("explanation", "")

                    st.markdown("## 📊 Prediction Result")

                    # -------------------------
                    # RESULT CARDS
                    # -------------------------
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric("Prediction", label)

                    with c2:
                        st.metric("Confidence", f"{confidence:.2f}")

                    with c3:
                        st.metric("Latency (s)", f"{latency:.3f}")

                    # -------------------------
                    # VISUAL RESULT
                    # -------------------------
                    if label == "Harmful":
                        st.error("🚨 Harmful Content Detected")
                    else:
                        st.success("✅ Content is Safe")

                    st.progress(int(confidence * 100))

                    # -------------------------
                    # EXPLANATION BLOCK
                    # -------------------------
                    st.markdown("### 🧠 Model Explanation")

                    if explanation:
                        st.info(explanation)
                    else:
                        st.warning("No explanation returned from model")

                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Connection error: {e}")

# =============================
# BATCH MODE (CSV)
# =============================
elif mode == "Batch CSV":

    st.markdown("## 📂 Batch Moderation")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        st.write("Preview:", df.head())

        if "content" not in df.columns:
            st.error("CSV must contain a 'content' column")
        else:

            if st.button("Run Batch Prediction"):

                results = []
                start = time.time()

                for text in df["content"].fillna("").astype(str):

                    try:
                        res = requests.post(API_URL, json={"content": text})
                        if res.status_code == 200:
                            results.append(res.json()["label"])
                        else:
                            results.append("Error")

                    except:
                        results.append("Error")

                df["prediction"] = results

                st.success("Batch prediction complete")
                st.dataframe(df)

                st.download_button(
                    "Download Results",
                    df.to_csv(index=False),
                    file_name="moderation_results.csv"
                )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("🚀 Pro Moderation System | FastAPI + XGBoost + Streamlit")