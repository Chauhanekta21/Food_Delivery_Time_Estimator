from pathlib import Path
import pickle

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

APP_COLORS = {
    "primary": "#5FA89D",
    "primary_dark": "#4F9187",
    "soft": "#EAF5F3",
    "bg": "#F7FAF9",
    "card": "#FFFFFF",
    "text": "#1F2933",
    "muted": "#52606D",
    "border": "#D9EAE7",
}

WEATHER_OPTIONS = ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
TRAFFIC_OPTIONS = ["High", "Low", "Medium"]
FEATURE_ORDER = [
    "Distance_km",
    "Preparation_Time_min",
    "Weather_Foggy",
    "Weather_Rainy",
    "Weather_Snowy",
    "Weather_Windy",
    "Traffic_Level_Low",
    "Traffic_Level_Medium",
]


def load_css():
    st.markdown(
        f"""
        <style>
            :root {{
                --primary: {APP_COLORS['primary']};
                --primary-dark: {APP_COLORS['primary_dark']};
                --soft: {APP_COLORS['soft']};
                --bg: {APP_COLORS['bg']};
                --card: {APP_COLORS['card']};
                --text: {APP_COLORS['text']};
                --muted: {APP_COLORS['muted']};
                --border: {APP_COLORS['border']};
            }}
            .stApp {{
                background: var(--bg);
                color: var(--text);
            }}
            .app-title {{
                background-image: linear-gradient(90deg, #244B67 0%, #4F9187 55%, #79C7BA 100%) !important;
                -webkit-background-clip: text !important;
                background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                color: transparent !important;
                display: inline-block;
                font-size: 2.35rem;
                font-weight: 700;
                line-height: 1.2;
                margin: 0 0 0.35rem;
            }}
            .subheading {{
                color: #39464D !important;
                margin-top: 0.65rem !important;
                margin-bottom: 0.65rem !important;
            }}
            [data-testid="stSidebar"] {{
                background: #172B2A;
                border-right: 1px solid #294341;
            }}
            [data-testid="stSidebar"] * {{
                color: #F4FAF8;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] input {{
                background: #F7FAF9;
                color: #1F2933;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] * {{
                color: #1F2933;
            }}
            [data-testid="stSidebar"] .stButton > button,
            [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {{
                background: var(--primary);
                color: white;
                border: none;
                margin-top: 0.8rem;
            }}
            .sidebar-heading {{
                color: #FFFFFF;
                font-size: 1.35rem;
                font-weight: 700;
                margin: 0.5rem 0 1.5rem;
            }}
            .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}
            .app-shell {{
                max-width: 920px;
                margin: 0 auto;
            }}
            .section-card {{
                background: rgba(255,255,255,0.92);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1.5rem 1.35rem;
                box-shadow: 0 8px 22px rgba(95, 168, 157, 0.06);
            }}
            .st-key-prediction-model-content {{
                background: #FFFFFF;
                border-radius: 18px;
                padding: 1.4rem 1.35rem 1.5rem;
                box-shadow: 0 6px 18px rgba(31, 41, 51, 0.07);
            }}
            .result-box {{
                background: linear-gradient(135deg, rgba(95,168,157,0.10), rgba(95,168,157,0.02));
                border: 1px solid rgba(95, 168, 157, 0.22);
                border-radius: 18px;
                padding: 1.5rem 1rem;
                text-align: center;
                margin-top: 0;
                margin-bottom: 0.65rem;
            }}
            .result-label {{
                font-size: 0.8rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--muted);
                font-weight: 600;
            }}
            .result-value {{
                font-size: 2.7rem;
                font-weight: 700;
                color: var(--primary-dark);
                margin-top: 0.25rem;
            }}
            .model-info-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 1rem;
                margin-bottom: 0.65rem;
            }}
            .model-info-box {{
                background: linear-gradient(135deg, rgba(95,168,157,0.10), rgba(95,168,157,0.02));
                border: 1px solid rgba(95, 168, 157, 0.22);
                border-radius: 18px;
                padding: 1rem;
                min-height: 5.25rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }}
            .model-info-label {{
                color: var(--primary-dark);
                font-weight: 600;
                line-height: 1.3;
            }}
            .model-info-value {{
                color: var(--muted);
                line-height: 1.3;
                margin-top: 0.35rem;
            }}
            .helper-text {{
                color: var(--muted);
                font-size: 0.88rem;
            }}
            .small-note {{
                margin-top: 1rem;
                color: var(--muted);
                font-size: 0.85rem;
                text-align: center;
            }}
            .stButton > button {{
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                padding: 0.7rem 1.2rem;
                width: 100%;
            }}
            .stButton > button:hover {{
                filter: brightness(1.02);
            }}
            .stButton > button:disabled {{
                background: #C9D9D6;
                color: #5A6B68;
            }}
            .expanderHeader {{
                color: var(--text);
                font-weight: 600;
            }}
            h1, h2, h3 {{
                color: var(--text);
            }}
            [data-testid="stExpander"] {{
                border: 1px solid var(--border);
                border-radius: 12px;
                background: var(--card);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    model_paths = [
        Path(__file__).resolve().parent / "delivery_time_model.pkl",
        Path(__file__).resolve().parent.parent / "delivery_time_model.pkl",
        Path(__file__).resolve().parent.parent / "streamlit_app" / "delivery_time_model.pkl",
    ]
    for path in model_paths:
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                try:
                    with open(path, "rb") as file:
                        return pickle.load(file)
                except Exception:
                    continue
    return None


@st.cache_data
def load_dataset():
    data_path = Path(__file__).resolve().parent.parent / "data" / "food_delivery_data.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return None


def build_feature_row(distance_km, preparation_time_min, weather, traffic_level):
    row = {
        "Distance_km": float(distance_km),
        "Preparation_Time_min": int(preparation_time_min),
        "Weather_Foggy": 1 if weather == "Foggy" else 0,
        "Weather_Rainy": 1 if weather == "Rainy" else 0,
        "Weather_Snowy": 1 if weather == "Snowy" else 0,
        "Weather_Windy": 1 if weather == "Windy" else 0,
        "Traffic_Level_Low": 1 if traffic_level == "Low" else 0,
        "Traffic_Level_Medium": 1 if traffic_level == "Medium" else 0,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict_delivery(model, distance_km, preparation_time_min, weather, traffic_level):
    features = build_feature_row(distance_km, preparation_time_min, weather, traffic_level)
    prediction = model.predict(features)[0]
    return round(float(prediction), 1)


def create_prediction_chart(model):
    df = load_dataset()
    if df is None:
        return None

    temp = df[["Distance_km", "Preparation_Time_min", "Weather", "Traffic_Level", "Delivery_Time_min"]].copy()
    temp = temp.dropna()
    temp = pd.get_dummies(temp, columns=["Weather", "Traffic_Level"], drop_first=True)

    for column in FEATURE_ORDER:
        if column not in temp.columns:
            temp[column] = 0

    temp = temp[FEATURE_ORDER + ["Delivery_Time_min"]]
    predictions = model.predict(temp[FEATURE_ORDER])

    chart_df = pd.DataFrame(
        {
            "Actual Delivery Time": temp["Delivery_Time_min"],
            "Predicted Delivery Time": predictions,
        }
    )

    fig = px.scatter(
        chart_df,
        x="Actual Delivery Time",
        y="Predicted Delivery Time",
        title="Actual vs Predicted Delivery Time",
        template="plotly_white",
        color_discrete_sequence=[APP_COLORS["primary"]],
        opacity=0.7,
    )
    fig.add_shape(
        type="line",
        x0=chart_df["Actual Delivery Time"].min(),
        x1=chart_df["Actual Delivery Time"].max(),
        y0=chart_df["Actual Delivery Time"].min(),
        y1=chart_df["Actual Delivery Time"].max(),
        line={"color": APP_COLORS["primary_dark"], "dash": "dash"},
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        title={
            "text": "Actual vs Predicted Delivery Time",
            "font": {"color": "#1F2933", "size": 20},
        },
        font={"color": "#1F2933", "size": 13},
        margin=dict(l=85, r=35, t=65, b=75),
        xaxis={
            "title": {
                "text": "Actual Delivery Time (minutes)",
                "font": {"color": "#1F2933", "size": 14},
            },
            "tickfont": {"color": "#1F2933", "size": 12},
            "showgrid": True,
            "gridcolor": "#C7D6D3",
            "linecolor": "#52606D",
            "zerolinecolor": "#52606D",
        },
        yaxis={
            "title": {
                "text": "Predicted Delivery Time (minutes)",
                "font": {"color": "#1F2933", "size": 14},
            },
            "tickfont": {"color": "#1F2933", "size": 12},
            "showgrid": True,
            "gridcolor": "#C7D6D3",
            "linecolor": "#52606D",
            "zerolinecolor": "#52606D",
        },
        height=500,
    )
    return fig


def render_sidebar(model):
    if "predicted_minutes" not in st.session_state:
        st.session_state.predicted_minutes = predict_delivery(model, 8.5, 12, "Clear", "Medium")

    with st.sidebar:
        st.markdown('<div class="sidebar-heading">Delivery Details</div>', unsafe_allow_html=True)
        st.caption("Enter the details needed for the prediction.")

        with st.form("delivery_form"):
            distance = st.number_input(
                "Distance (km)",
                min_value=0.5,
                max_value=25.0,
                value=8.5,
                step=0.5,
                help="Distance from the restaurant to the destination in kilometers.",
            )
            preparation_time = st.number_input(
                "Preparation time (min)",
                min_value=1,
                max_value=40,
                value=12,
                step=1,
                help="Time taken to prepare the order in minutes.",
            )
            weather = st.selectbox("Weather", options=WEATHER_OPTIONS, index=0)
            traffic_level = st.selectbox("Traffic", options=TRAFFIC_OPTIONS, index=1)
            submit = st.form_submit_button("Estimate Delivery Time", use_container_width=True)

    if submit and distance > 0 and preparation_time > 0:
        st.session_state.predicted_minutes = predict_delivery(
            model, distance, preparation_time, weather, traffic_level
        )


def render_prediction():
    predicted_minutes = st.session_state.get("predicted_minutes")
    st.markdown(
        """
        <div class="result-box">
            <div class="result-label">The order is estimated to arrive in approximately</div>
            <div class="result-value">{} minutes</div>
        </div>
        """.format(f"{predicted_minutes:.1f}"),
        unsafe_allow_html=True,
    )


def render_about_model(model):
    st.markdown('<h2 class="subheading">About the Model</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="model-info-grid">
            <div class="model-info-box">
                <div class="model-info-label">Model</div>
                <div class="model-info-value">Linear Regression</div>
            </div>
            <div class="model-info-box">
                <div class="model-info-label">Model accuracy</div>
                <div class="model-info-value">81.6%</div>
            </div>
            <div class="model-info-box">
                <div class="model-info-label">Average prediction error</div>
                <div class="model-info-value">6.25 minutes</div>
            </div>
            <div class="model-info-box">
                <div class="model-info-label">Tools used</div>
                <div class="model-info-value">Python, Scikit-learn, Streamlit</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = create_prediction_chart(model)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Food Delivery Time Estimator", layout="centered")
    load_css()

    model = load_model()
    if model is None:
        st.warning("The model file was not found. Please place delivery_time_model.pkl in the project root before running the app.")
        st.stop()

    render_sidebar(model)

    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">Food Delivery Time Estimator</h1>', unsafe_allow_html=True)
    st.caption("Estimate delivery time using the trained Linear Regression model.")
    with st.container(key="prediction-model-content"):
        st.markdown('<h2 class="subheading">Prediction result</h2>', unsafe_allow_html=True)
        render_prediction()
        render_about_model(model)

    st.markdown("---")
    st.markdown(
        "<div class='small-note'>Note: This prediction is based on the trained Linear Regression model for this dataset only.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
