import sys
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk


# -------------------------------------------------------
# Project imports
# -------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.predict import predict_from_lat_lon
from src.load_artifacts import load_data


# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="TrafficPulse | 3D Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------------------------------------------
# Optional auto refresh
# -------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except Exception:
    AUTO_REFRESH_AVAILABLE = False


# -------------------------------------------------------
# CSS: animated advanced UI
# -------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 20% 20%, rgba(0, 255, 170, 0.12), transparent 25%),
            radial-gradient(circle at 80% 10%, rgba(0, 140, 255, 0.15), transparent 30%),
            radial-gradient(circle at 50% 90%, rgba(255, 77, 77, 0.10), transparent 35%),
            linear-gradient(135deg, #050816 0%, #0b1020 45%, #101624 100%);
        color: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 18, 35, 0.98), rgba(5, 8, 22, 0.98));
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero-card {
        position: relative;
        padding: 28px 32px;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(0, 255, 170, 0.16), rgba(0, 140, 255, 0.10));
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 0 35px rgba(0, 255, 170, 0.08);
        overflow: hidden;
        margin-bottom: 20px;
    }

    .hero-card::before {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: rgba(0,255,170,0.18);
        top: -60px;
        right: -50px;
        animation: pulse 3s infinite ease-in-out;
    }

    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.45; }
        50% { transform: scale(1.15); opacity: 0.85; }
        100% { transform: scale(0.9); opacity: 0.45; }
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #00ffaa, #00aaff, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 16px;
        color: rgba(255,255,255,0.78);
        max-width: 850px;
        line-height: 1.7;
    }

    .metric-card {
        padding: 20px;
        border-radius: 22px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        min-height: 135px;
        transition: 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0,255,170,0.45);
        box-shadow: 0 14px 40px rgba(0,255,170,0.12);
    }

    .metric-label {
        color: rgba(255,255,255,0.65);
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 800;
        margin-top: 8px;
        color: #ffffff;
    }

    .metric-help {
        color: rgba(255,255,255,0.58);
        font-size: 13px;
        margin-top: 8px;
    }

    .status-low {
        color: #00ffaa;
        text-shadow: 0 0 15px rgba(0,255,170,0.35);
    }

    .status-medium {
        color: #ffd166;
        text-shadow: 0 0 15px rgba(255,209,102,0.35);
    }

    .status-high {
        color: #ff4d6d;
        text-shadow: 0 0 15px rgba(255,77,109,0.35);
    }

    .glass-box {
        padding: 18px 22px;
        border-radius: 20px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 14px;
    }

    .small-muted {
        font-size: 13px;
        color: rgba(255,255,255,0.60);
    }

    div[data-testid="stMetricValue"] {
        color: white;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.08);
        border-radius: 14px;
        color: white;
        padding: 12px 18px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,255,170,0.22), rgba(0,140,255,0.20));
        border: 1px solid rgba(0,255,170,0.40);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
HORIZONS = ["5min", "15min", "30min"]


def normalize_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "latitude" not in df.columns and "latitude_x" in df.columns:
        df["latitude"] = df["latitude_x"]

    if "longitude" not in df.columns and "longitude_x" in df.columns:
        df["longitude"] = df["longitude_x"]

    return df


def clean_list_string(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        value = (
            value.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )

    return value


def get_congestion_level(speed: float) -> str:
    if speed >= 50:
        return "Low"
    elif speed >= 30:
        return "Medium"
    return "High"


def congestion_color(level: str):
    if level == "Low":
        return [0, 255, 170, 185]
    elif level == "Medium":
        return [255, 209, 102, 205]
    return [255, 77, 109, 230]


def congestion_css_class(level: str) -> str:
    if level == "Low":
        return "status-low"
    elif level == "Medium":
        return "status-medium"
    return "status-high"


def extract_predicted_speed(result: dict) -> float:
    if "predicted_speed_mph" in result:
        return float(result["predicted_speed_mph"])
    if "predicted_speed" in result:
        return float(result["predicted_speed"])
    return np.nan


@st.cache_data(show_spinner=False)
def cached_load_data():
    df = load_data()
    df = normalize_coordinate_columns(df)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "sensor_id" in df.columns:
        df["sensor_id"] = df["sensor_id"].astype(str)

    return df


@st.cache_data(show_spinner=False)
def cached_prediction(lat: float, lon: float, horizon: str):
    return predict_from_lat_lon(float(lat), float(lon), horizon)


def create_latest_sensor_map_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["sensor_id", "timestamp", "speed", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns for map: {missing}")

    latest = (
        df.sort_values("timestamp")
        .groupby("sensor_id", as_index=False)
        .tail(1)
        .copy()
    )

    latest["congestion_level"] = latest["speed"].apply(get_congestion_level)
    latest["color"] = latest["congestion_level"].apply(congestion_color)

    latest["congestion_score"] = (70 - latest["speed"]).clip(lower=1)
    latest["elevation"] = latest["congestion_score"] * 35

    return latest


def create_prediction_gauge(speed: float, level: str):
    if level == "Low":
        bar_color = "#00ffaa"
    elif level == "Medium":
        bar_color = "#ffd166"
    else:
        bar_color = "#ff4d6d"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=speed,
            number={"suffix": " mph", "font": {"size": 36, "color": "white"}},
            gauge={
                "axis": {"range": [0, 80], "tickcolor": "white"},
                "bar": {"color": bar_color},
                "bgcolor": "rgba(255,255,255,0.08)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.25)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(255,77,109,0.25)"},
                    {"range": [30, 50], "color": "rgba(255,209,102,0.25)"},
                    {"range": [50, 80], "color": "rgba(0,255,170,0.22)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": speed,
                },
            },
            title={
                "text": f"Predicted Speed | {level} Congestion",
                "font": {"size": 18, "color": "white"},
            },
        )
    )

    fig.update_layout(
        height=310,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=10),
    )

    return fig


def create_la_congestion_map(
    map_df: pd.DataFrame,
    nearest_sensor_id: str,
    input_lat: float,
    input_lon: float,
):
    """
    LA-centered congestion map.
    Shows latest observed congestion level for all METR-LA sensors.
    Highlights the nearest sensor used by the model and the user's input point.
    """

    plot_df = map_df.copy()
    plot_df["sensor_id"] = plot_df["sensor_id"].astype(str)

    nearest_df = plot_df[
        plot_df["sensor_id"] == str(nearest_sensor_id)
    ].copy()

    input_df = pd.DataFrame(
        {
            "latitude": [input_lat],
            "longitude": [input_lon],
            "label": ["Input location"],
        }
    )

    # Fixed LA center because model is trained on METR-LA
    la_center_lat = 34.052235
    la_center_lon = -118.243683

    sensor_layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[longitude, latitude]",
        get_radius=180,
        get_fill_color="color",
        get_line_color=[255, 255, 255, 80],
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=True,
        auto_highlight=True,
    )

    nearest_sensor_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nearest_df,
        get_position="[longitude, latitude]",
        get_radius=520,
        get_fill_color=[255, 255, 255, 40],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=4,
        stroked=True,
        filled=True,
        pickable=True,
    )

    input_location_layer = pdk.Layer(
        "ScatterplotLayer",
        data=input_df,
        get_position="[longitude, latitude]",
        get_radius=350,
        get_fill_color=[0, 140, 255, 210],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=3,
        stroked=True,
        filled=True,
        pickable=True,
    )

    tooltip = {
        "html": """
        <b>Sensor:</b> {sensor_id}<br/>
        <b>Latest speed:</b> {speed} mph<br/>
        <b>Congestion:</b> {congestion_level}<br/>
        """,
        "style": {
            "backgroundColor": "rgba(10, 18, 35, 0.95)",
            "color": "white",
            "border": "1px solid rgba(0,255,170,0.6)",
            "borderRadius": "10px",
            "padding": "10px",
        },
    }

    deck = pdk.Deck(
        layers=[
            sensor_layer,
            nearest_sensor_layer,
            input_location_layer,
        ],
        initial_view_state=pdk.ViewState(
            latitude=la_center_lat,
            longitude=la_center_lon,
            zoom=10.2,
            pitch=0,
            bearing=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip=tooltip,
    )

    return deck


def show_congestion_legend():
    st.markdown(
        """
        <div class="glass-box">
            <b>Congestion Map Legend</b><br><br>
            <span style="color:#00ffaa;">●</span> Low congestion: predicted/observed speed ≥ 50 mph<br>
            <span style="color:#ffd166;">●</span> Medium congestion: speed between 30 and 50 mph<br>
            <span style="color:#ff4d6d;">●</span> High congestion: speed below 30 mph<br>
            <span style="color:#ffffff;">◎</span> White ring: nearest METR-LA sensor used for prediction<br>
            <span style="color:#008cff;">●</span> Blue marker: user-entered coordinate
        </div>
        """,
        unsafe_allow_html=True,
    )

def create_congestion_distribution_chart(map_df: pd.DataFrame):
    congestion_counts = (
        map_df["congestion_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .reset_index()
    )

    congestion_counts.columns = ["congestion_level", "sensor_count"]

    fig = px.bar(
        congestion_counts,
        x="congestion_level",
        y="sensor_count",
        color="congestion_level",
        text="sensor_count",
        title="Current Congestion Distribution Across METR-LA Sensors",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        height=340,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis_title="Congestion level",
        yaxis_title="Number of sensors",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )

    return fig

def create_3d_scatter(map_df: pd.DataFrame, nearest_sensor_id: str):
    sample_df = map_df.copy()

    if len(sample_df) > 1200:
        sample_df = sample_df.sample(1200, random_state=42)

    sample_df["selected"] = np.where(
        sample_df["sensor_id"].astype(str) == str(nearest_sensor_id),
        "Nearest sensor",
        "Other sensors",
    )

    fig = px.scatter_3d(
        sample_df,
        x="longitude",
        y="latitude",
        z="speed",
        color="congestion_level",
        size="congestion_score",
        hover_data=["sensor_id", "speed", "congestion_level"],
        title="3D Speed Surface Across METR-LA Sensors",
    )

    fig.update_layout(
        height=650,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Speed mph",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def create_speed_trend(df: pd.DataFrame, sensor_id: str, prediction_results: dict):
    sensor_df = (
        df[df["sensor_id"].astype(str) == str(sensor_id)]
        .sort_values("timestamp")
        .tail(288)
        .copy()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sensor_df["timestamp"],
            y=sensor_df["speed"],
            mode="lines",
            name="Historical speed",
            line=dict(width=3),
        )
    )

    if not sensor_df.empty:
        latest_time = sensor_df["timestamp"].max()

        future_times = []
        future_speeds = []

        for horizon, minutes in [("5min", 5), ("15min", 15), ("30min", 30)]:
            if horizon in prediction_results:
                future_times.append(latest_time + timedelta(minutes=minutes))
                future_speeds.append(extract_predicted_speed(prediction_results[horizon]))

        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=future_speeds,
                mode="markers+lines",
                name="Predicted speed",
                marker=dict(size=11, symbol="diamond"),
                line=dict(width=3, dash="dash"),
            )
        )

    fig.update_layout(
        height=420,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        title=f"Recent Speed Trend for Sensor {sensor_id}",
        xaxis_title="Time",
        yaxis_title="Speed mph",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_horizon_bar(prediction_results: dict):
    rows = []

    for horizon in HORIZONS:
        if horizon in prediction_results:
            speed = extract_predicted_speed(prediction_results[horizon])
            rows.append(
                {
                    "horizon": horizon,
                    "predicted_speed": speed,
                    "congestion_level": get_congestion_level(speed),
                }
            )

    chart_df = pd.DataFrame(rows)

    fig = px.bar(
        chart_df,
        x="horizon",
        y="predicted_speed",
        color="congestion_level",
        text="predicted_speed",
        title="Prediction Comparison Across Horizons",
    )

    fig.update_traces(texttemplate="%{text:.2f} mph", textposition="outside")

    fig.update_layout(
        height=360,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis_title="Forecast horizon",
        yaxis_title="Predicted speed mph",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def make_metric_card(label, value, help_text="", css_class=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css_class}">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_selected_sensor_latest_row(df: pd.DataFrame, sensor_id: str) -> pd.Series:
    sensor_rows = (
        df[df["sensor_id"].astype(str) == str(sensor_id)]
        .sort_values("timestamp")
        .copy()
    )

    if sensor_rows.empty:
        return pd.Series(dtype=object)

    return sensor_rows.iloc[-1]


def display_road_weather_cards(latest_row: pd.Series):
    road_cols = ["highway", "length", "lanes", "maxspeed", "oneway", "distance_to_road"]
    weather_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "weather_code",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
    ]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🛣️ Road segment context")
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)

        for col in road_cols:
            if col in latest_row.index:
                value = clean_list_string(latest_row[col])
                st.write(f"**{col}:** {value}")

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("#### 🌦️ Weather context")
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)

        for col in weather_cols:
            if col in latest_row.index:
                st.write(f"**{col}:** {latest_row[col]}")

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------
# Load data
# -------------------------------------------------------
try:
    df = cached_load_data()
    map_df = create_latest_sensor_map_df(df)
except Exception as e:
    st.error("Failed to load project data.")
    st.exception(e)
    st.stop()


# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🚦 TrafficPulse 3D Intelligence</div>
        <div class="hero-subtitle">
            Advanced ML-powered traffic speed and congestion forecasting dashboard with 3D sensor visualization,
            weather-aware context, road-network features, and multi-horizon predictions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tab_input, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎛️ Input Panel",
        "🚀 Prediction Cockpit",
        "🗺️ LA Congestion Map",
        "📈 Trends & Horizons",
        "🧠 Context & Explainability",
        "📦 Data & Export",
    ]
)



# -------------------------------------------------------
# Input Panel Tab
# -------------------------------------------------------
with tab_input:
    st.markdown("## 🎛️ TrafficPulse Input Panel")
    st.caption(
        "This dashboard is currently built for Los Angeles only because the model is trained on METR-LA traffic sensor data."
    )

    st.markdown("---")



    st.markdown("### 📍 Enter Location")

    loc_col1, loc_col2 = st.columns(2)

    with loc_col1:
        lat_text = st.text_input(
            "Latitude",
            value="",
            placeholder="Example: 34.15497",
            key="lat_input_main",
        )

    with loc_col2:
        lon_text = st.text_input(
            "Longitude",
            value="",
            placeholder="Example: -118.31829",
            key="lon_input_main",
        )


    selected_horizon = st.selectbox(
    "Primary prediction horizon",
    HORIZONS,
    index=0,
    key="selected_horizon_main",
)

    st.info(
        "Enter latitude and longitude from the Los Angeles area. Predictions outside Los Angeles may be unreliable because the nearest-sensor matching will still map your point to an LA traffic sensor."
    )

    st.markdown("---")

    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        run_button = st.button(
            "🚀 Run Prediction",
            use_container_width=True,
            key="run_prediction_main",
        )

    with control_col2:
        auto_refresh = st.toggle(
            "Live simulation refresh",
            value=False,
            key="auto_refresh_main",
        )

    with control_col3:
        refresh_seconds = st.slider(
            "Refresh interval",
            min_value=5,
            max_value=60,
            value=15,
            step=5,
            key="refresh_seconds_main",
            disabled=not auto_refresh,
        )

    if auto_refresh:
        if AUTO_REFRESH_AVAILABLE:
            st_autorefresh(
                interval=refresh_seconds * 1000,
                key="trafficpulse_refresh_main",
            )
        else:
            st.warning(
                "Install streamlit-autorefresh for live refresh: pip install streamlit-autorefresh"
            )

    st.markdown("### ⚙️ Advanced Display Options")

    option_col1, option_col2 = st.columns(2)

    with option_col1:
        show_debug = st.toggle(
            "Show debug information",
            value=False,
            key="show_debug_main",
        )

    with option_col2:
        show_data_preview = st.toggle(
            "Show data preview",
            value=False,
            key="show_data_preview_main",
        )



# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
lat = None
lon = None

if run_button:
    try:
        lat = float(lat_text.strip())
        lon = float(lon_text.strip())
    except ValueError:
        with tab_input:
            st.error("Please enter valid numeric latitude and longitude values.")
        st.stop()

    with st.spinner("Running ML prediction for all horizons..."):
        prediction_results = {}

        for horizon in HORIZONS:
            try:
                prediction_results[horizon] = cached_prediction(lat, lon, horizon)
            except Exception as e:
                st.error(f"Prediction failed for {horizon}.")
                st.exception(e)
                st.stop()

        st.session_state["prediction_results"] = prediction_results
        st.session_state["lat"] = lat
        st.session_state["lon"] = lon

if "prediction_results" not in st.session_state:
    with tab_input:
        st.warning("Enter latitude and longitude, then click **Run Prediction** to start.")
    st.stop()

prediction_results = st.session_state["prediction_results"]

lat = st.session_state["lat"]
lon = st.session_state["lon"]

primary_result = prediction_results[selected_horizon]
primary_speed = extract_predicted_speed(primary_result)
primary_level = get_congestion_level(primary_speed)
primary_css = congestion_css_class(primary_level)
nearest_sensor_id = str(primary_result["nearest_sensor_id"])
nearest_distance = float(primary_result["distance_km"])

latest_row = get_selected_sensor_latest_row(df, nearest_sensor_id)


# -------------------------------------------------------
# Top metrics
# -------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    make_metric_card(
        "Predicted speed",
        f"{primary_speed:.2f} mph",
        f"Horizon: {selected_horizon}",
        primary_css,
    )

with m2:
    make_metric_card(
        "Congestion level",
        primary_level,
        "Based on predicted speed threshold",
        primary_css,
    )

with m3:
    make_metric_card(
        "Nearest sensor",
        nearest_sensor_id,
        f"{nearest_distance:.3f} km from input",
    )

with m4:
    if not latest_row.empty and "speed" in latest_row.index:
        latest_speed = float(latest_row["speed"])
        make_metric_card(
            "Latest observed speed",
            f"{latest_speed:.2f} mph",
            "Most recent value in dataset",
        )
    else:
        make_metric_card("Latest observed speed", "N/A", "No recent row found")


if nearest_distance > 30:
    st.warning(
        f"The input point is {nearest_distance:.2f} km away from the nearest METR-LA sensor. "
        "Predictions are most meaningful near Los Angeles sensor locations."
    )










# -------------------------------------------------------
# Tab 1: Prediction cockpit
# -------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.plotly_chart(
            create_prediction_gauge(primary_speed, primary_level),
            use_container_width=True,
        )

    with c2:
        st.plotly_chart(
            create_horizon_bar(prediction_results),
            use_container_width=True,
        )

    st.markdown("### Forecast summary")

    summary_cols = st.columns(3)

    for idx, horizon in enumerate(HORIZONS):
        result = prediction_results[horizon]
        speed = extract_predicted_speed(result)
        level = get_congestion_level(speed)

        with summary_cols[idx]:
            make_metric_card(
                f"{horizon} forecast",
                f"{speed:.2f} mph",
                f"{level} congestion",
                congestion_css_class(level),
            )


# -------------------------------------------------------
# Tab 2: LA Congestion Map
# -------------------------------------------------------
with tab2:
    st.markdown("### 🗺️ Los Angeles Congestion Map")

    st.caption(
        "This map is centered around Los Angeles because the model is trained on METR-LA traffic sensor data."
    )

    show_congestion_legend()

    try:
        st.pydeck_chart(
            create_la_congestion_map(
                map_df=map_df,
                nearest_sensor_id=nearest_sensor_id,
                input_lat=lat,
                input_lon=lon,
            ),
            use_container_width=True,
        )
    except Exception as e:
        st.warning("Map failed to render.")
        st.exception(e)

    st.markdown("### 📊 Sensor congestion summary")

    st.plotly_chart(
        create_congestion_distribution_chart(map_df),
        use_container_width=True,
    )

# -------------------------------------------------------
# Tab 3: Trends
# -------------------------------------------------------
with tab3:
    st.markdown("### 📈 Historical trend + predicted future points")

    st.plotly_chart(
        create_speed_trend(df, nearest_sensor_id, prediction_results),
        use_container_width=True,
    )

    st.markdown("### Sensor-level quick stats")

    selected_sensor_history = df[df["sensor_id"].astype(str) == nearest_sensor_id].copy()

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric("Mean speed", f"{selected_sensor_history['speed'].mean():.2f} mph")

    with s2:
        st.metric("Min speed", f"{selected_sensor_history['speed'].min():.2f} mph")

    with s3:
        st.metric("Max speed", f"{selected_sensor_history['speed'].max():.2f} mph")

    with s4:
        st.metric("Records", f"{len(selected_sensor_history):,}")


# -------------------------------------------------------
# Tab 4: Context
# -------------------------------------------------------
with tab4:
    st.markdown("### 🧠 Prediction context")

    if latest_row.empty:
        st.warning("No latest row found for the nearest sensor.")
    else:
        display_road_weather_cards(latest_row)

    st.markdown("### Simple rule-based explanation")

    explanations = []

    if primary_speed < 30:
        explanations.append("Predicted speed is below 30 mph, so the model output is classified as high congestion.")
    elif primary_speed < 50:
        explanations.append("Predicted speed is between 30 and 50 mph, so traffic is classified as medium congestion.")
    else:
        explanations.append("Predicted speed is above 50 mph, so traffic is classified as low congestion.")

    if "rain" in latest_row.index and float(latest_row["rain"]) > 0:
        explanations.append("Rain is present in the weather data, which can contribute to slower traffic.")

    if "precipitation" in latest_row.index and float(latest_row["precipitation"]) > 0:
        explanations.append("Precipitation is non-zero, indicating weather-related driving friction.")

    if "distance_to_road" in latest_row.index:
        explanations.append(
            f"The selected sensor is approximately {float(latest_row['distance_to_road']):.2f} meters from the matched OSM road segment."
        )

    if "highway" in latest_row.index:
        explanations.append(f"The matched OSM road type is {clean_list_string(latest_row['highway'])}.")

    for exp in explanations:
        st.markdown(
            f"""
            <div class="glass-box">
                ✅ {exp}
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------
# Tab 5: Data and export
# -------------------------------------------------------
with tab5:
    st.markdown("### 📦 Prediction output")

    export_payload = {
        "input": {
            "latitude": lat,
            "longitude": lon,
            "selected_horizon": selected_horizon,
        },
        "nearest_sensor": {
            "sensor_id": nearest_sensor_id,
            "distance_km": nearest_distance,
        },
        "predictions": prediction_results,
    }

    st.json(export_payload)

    st.download_button(
        label="⬇️ Download prediction JSON",
        data=json.dumps(export_payload, indent=4),
        file_name="trafficpulse_prediction.json",
        mime="application/json",
        use_container_width=True,
    )

    if show_data_preview:
        st.markdown("### Latest sensor map data")
        st.dataframe(map_df.head(100), use_container_width=True)

        st.markdown("### Selected sensor latest row")
        st.dataframe(pd.DataFrame([latest_row]), use_container_width=True)

    if show_debug:
        st.markdown("### Debug information")
        st.write("Data columns:")
        st.write(df.columns.tolist())
        st.write("Primary result:")
        st.write(primary_result)
        st.write("Nearest sensor latest row:")
        st.write(latest_row)


# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown(
    """
    <div class="small-muted" style="text-align:center; margin-top: 30px;">
        TrafficPulse dashboard powered by METR-LA traffic data, OSM road features, weather features, and ML forecasting.
    </div>
    """,
    unsafe_allow_html=True,
)