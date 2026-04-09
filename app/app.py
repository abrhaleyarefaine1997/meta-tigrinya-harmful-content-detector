import streamlit as st
import pandas as pd

# --- Keyword list ---
context_markers = [
    "ደማት", "ነገራትኩም ግጉይን ሰይጣናውን ዩ", "ይተርፍ", "በሎ", "ምስንቱ ኣነ", 
    "ድሮም አስኩም ዶ ዘይትረብሑ", "ምጉሓፎም", "ህዝቢታት ብጣዕሚ ቅርሕንቲ አለዎም",
    "ቡዳ", "ጠቢብ", "በላዕቲ ሰብ አዮም", "ትኳን ዝኮነ ህዝቢ", "ደማዊ ጦርነት",
    "ዲቃላ", "ይ ዒፍ ቋንቋ ዝዛረቡ ህዝቢ", "ክርችሽኑ አዩ ዘለዎም", "ዘይሰብ",
    "ንሰብ ዘይመስል", "ቀታሊ አዩ", "ንህዝብና ቀተልቱ መራሕትና አዮም", "ናብ ውግእ",
    "ዘርኣዊ ፅንተት", "ኣምባገነንነት", "ዘቅትል ዘረሽንን", "ወረርቲ ሓይልታት ዝሞቱ",
    "ህበይ", "ወዓግ", "ተመን", "ጨምላቕ", "ግብረ እከይ", "ኢዚ ሰብ ዘይኮነ",
    "ነዚኦም ከመይ ኢልና ንሓዝነሎም", "ሰይጣን", "ቐታ", "ጦርነት", "ኩናት", "ውግእ",
    "ወዲ ሻርሙጣ", "ወዲ ዓጣሪት", "ፅናሕ ከርእየካ እየ ግዜኻ ተፀበ",
    "ሃይማኖትኩም ትክክለኛ ኣይኮነን", "አድጊ"
]

# --- Load your data (replace with your actual path) ---
# meta_data = pd.read_csv("meta_data.csv")

# For demonstration, use fake data if not loading real
meta_data = pd.read_csv("/Users/a1234/meta_tigrinya_dataset_cleaned.csv")

# --- Apply Rule-Based Detection ---
def detect_marker(content):
    for marker in context_markers:
        if pd.notna(content) and marker in content:
            return marker
    return None

meta_data["matched_keyword"] = meta_data["content"].apply(detect_marker)
meta_data["predicted_label"] = meta_data["matched_keyword"].apply(lambda x: "Harmful" if x else "Neutral")

# --- Metrics ---
accuracy = (meta_data["label"] == meta_data["predicted_label"]).mean()
conf_matrix = pd.crosstab(meta_data["label"], meta_data["predicted_label"], rownames=["Actual"], colnames=["Predicted"])

# --- Streamlit App ---
st.set_page_config(page_title="Tigrinya Harmful Meta Post Detector", layout="wide")

st.markdown("<h1 style='color:#4b0082;'>🛡️ Tigrinya Harmful Meta Post Detector</h1>", unsafe_allow_html=True)
st.markdown("🚨 Rule-based detection for harmful content in Tigrinya posts.")

# --- Text Input ---
post = st.text_area("✍️ Enter a Tigrinya Post", placeholder="ምሳናይ ፅሑፍ...", height=150)

if st.button("🔍 Analyze"):
    detected_keywords = [kw for kw in context_markers if kw in post]
    num_keywords = len(detected_keywords)
    label = "🟥 Harmful" if num_keywords > 0 else "🟩 Neutral"
    
    st.markdown(f"### ✅ Prediction: {label}")
    st.markdown(f"**📝 Matched Keywords:** {', '.join(detected_keywords) if detected_keywords else 'None'}")
    st.markdown(f"**🔢 Keyword Count:** {num_keywords}")
    st.markdown(f"**📈 Model Accuracy:** {accuracy:.2%}")

# --- Confusion Matrix ---
st.markdown("### 📊 Confusion Matrix")
st.dataframe(conf_matrix, use_container_width=True)
