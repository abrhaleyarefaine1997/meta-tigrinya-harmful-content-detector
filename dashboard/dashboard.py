import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# ========================
# 🔍 Load and Process Data
# ========================

# --- Harmful context markers ---
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

# --- Load dataset ---
meta_data = pd.read_csv("/Users/a1234/meta_tigrinya_dataset_cleaned.csv")

# --- Rule-based classification ---
def detect_marker(content):
    for marker in context_markers:
        if pd.notna(content) and marker in content:
            return marker
    return None

meta_data["matched_keyword"] = meta_data["content"].apply(detect_marker)
meta_data["predicted_label"] = meta_data["matched_keyword"].apply(lambda x: "Harmful" if x else "Neutral")

# --- Evaluation metrics ---
accuracy = (meta_data["label"] == meta_data["predicted_label"]).mean()
conf_matrix = pd.crosstab(meta_data["label"], meta_data["predicted_label"], rownames=["Actual"], colnames=["Predicted"])

# ========================
# 🎨 Dash App
# ========================

app = dash.Dash(__name__)
app.title = "Tigrinya Harmful Post Detector"

app.layout = html.Div([
    html.H1("🛡️ Tigrinya Harmful Meta Post Detector", style={'color': '#4b0082'}),
    html.P("🚨 Rule-based detection for harmful content in Tigrinya posts."),

    dcc.Textarea(
        id='post-input',
        placeholder="✍️ Enter a Tigrinya Post...",
        style={'width': '100%', 'height': 150, 'fontSize': 16}
    ),
    html.Button("🔍 Analyze", id="analyze-btn", style={'marginTop': '10px'}),

    html.Div(id="result-box", style={'marginTop': '20px'}),

    html.H3("📈 Model Accuracy", style={'marginTop': '30px'}),
    html.Div(f"{accuracy:.2%}", style={'fontSize': '20px', 'color': 'green'}),

    html.H3("📊 Confusion Matrix"),
    dcc.Graph(
        id="conf-matrix-graph",
        figure=ff.create_annotated_heatmap(
            z=conf_matrix.values,
            x=conf_matrix.columns.tolist(),
            y=conf_matrix.index.tolist(),
            colorscale="Blues"
        )
    ),
    
    html.H3("📃 Top Posts Detected as Harmful"),
    dcc.Dropdown(
        id="harmful-posts-dropdown",
        options=[{"label": row["content"][:80] + "...", "value": row["content"]} 
                 for _, row in meta_data[meta_data["predicted_label"]=="Harmful"].head(10).iterrows()],
        placeholder="Select a sample harmful post"
    ),
    html.Div(id="selected-harmful-post", style={'marginTop': '10px', 'fontStyle': 'italic', 'color': '#b30000'})
])     

# ========================
# 🔁 Callbacks
# ========================

@app.callback(
    Output("result-box", "children"),
    Input("analyze-btn", "n_clicks"),
    State("post-input", "value")
)
def classify_post(n_clicks, post):
    if not post:
        return ""
    
    detected_keywords = [kw for kw in context_markers if kw in post]
    label = "🟥 Harmful" if detected_keywords else "🟩 Neutral"
    return html.Div([
        html.P(f"✅ Prediction: {label}", style={'fontSize': '18px'}),
        html.P(f"🧩 Matched Keywords: {', '.join(detected_keywords) if detected_keywords else 'None'}"),
        html.P(f"🔢 Keyword Count: {len(detected_keywords)}")
    ])

@app.callback(
    Output("selected-harmful-post", "children"),
    Input("harmful-posts-dropdown", "value")
)
def display_selected_post(post):
    if post:
        return f"📝 {post}"
    return ""

# ========================
# 🚀 Run App
# ========================

if __name__ == "__main__":
    app.run(debug=True)
