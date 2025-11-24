import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime

# =====================================================
# Load Model and Scaler
# =====================================================
@st.cache_resource
def load_model():
    with open("xgb_hourly_demand.pkl", "rb") as file:
        model_package = pickle.load(file)
    return model_package

model_package = load_model()
model = model_package["model"]
scaler = model_package["scaler"]
feature_columns = model_package["feature_columns"]

# =====================================================
# Helper Functions
# =====================================================
def create_features(date, hour, temp, humidity, precipitation, windspeed, cloudcover):
    """Generate feature row with cyclic encodings."""
    dt = pd.to_datetime(date)

    # Compute cyclical encodings
    hour_sin = np.sin(hour / 24 * 2 * np.pi)
    hour_cos = np.cos(hour / 24 * 2 * np.pi)
    month_sin = np.sin(dt.month / 12 * 2 * np.pi)
    month_cos = np.cos(dt.month / 12 * 2 * np.pi)
    dow_sin = np.sin(dt.weekday() / 7 * 2 * np.pi)
    dow_cos = np.cos(dt.weekday() / 7 * 2 * np.pi)
    season = ((dt.month % 12 + 3) // 3)  # 1–4
    season_sin = np.sin(season / 4 * 2 * np.pi)
    season_cos = np.cos(season / 4 * 2 * np.pi)
    is_weekend = 1 if dt.weekday() >= 5 else 0

    row = {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "season_sin": season_sin,
        "season_cos": season_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_weekend": is_weekend,
        "temperature_2m": temp,
        "relative_humidity_2m": humidity,
        "precipitation": precipitation,
        "wind_speed_10m": windspeed,
        "cloudcover": cloudcover,
    }

    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def predict_hourly_demand(input_df):
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    return prediction


def predict_full_day(date, temp, humidity, precipitation, windspeed, cloudcover):
    """Predict demand for all 24 hours of the selected day."""
    records = []
    for hour in range(24):
        df_hour = create_features(date, hour, temp, humidity, precipitation, windspeed, cloudcover)
        pred = predict_hourly_demand(df_hour)
        records.append({"Hour": hour, "Predicted Demand": pred})
    return pd.DataFrame(records)

# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(page_title="🚲 Bike Demand Prediction", layout="wide")

st.sidebar.title("Predicting Hourly Bike Rental Demand")
st.sidebar.markdown("Use weather and time inputs to forecast hourly rental demand.")

# --- Inputs ---
st.subheader("Select Date & Hour")
col1, col2 = st.columns(2)
with col1:
    date = st.date_input("📅 Select Date", datetime(2025, 11, 24))
with col2:
    hour = st.slider("⏰ Hour of the Day (0–23)", 0, 23, 17)

st.subheader("Enter Weather Data 🌦")
temp = st.number_input("Temperature (°C)", value=10.0, step=0.5)
humidity = st.number_input("Humidity (%)", value=46.0, step=1.0)
precip = st.number_input("Precipitation (mm)", value=0.0, step=0.1)
windspeed = st.number_input("Wind Speed (km/h)", value=9.4, step=0.5)
cloudcover = st.number_input("Cloudcover (%)", value=45.0, step=1.0)

# --- Predict Button ---
if st.button("🔮 Predict Demand"):
    input_df = create_features(date, hour, temp, humidity, precip, windspeed, cloudcover)
    prediction = predict_hourly_demand(input_df)
    st.success(f"🚲 **Predicted Hourly Demand at {hour}:00 → {prediction:.0f} bikes/hour**")

    # Predict entire day
    daily_preds = predict_full_day(date, temp, humidity, precip, windspeed, cloudcover)

    # Highlight selected hour
    daily_preds["color"] = np.where(daily_preds["Hour"] == hour, "Selected Hour", "Other Hours")

    fig = px.bar(
        daily_preds,
        x="Hour",
        y="Predicted Demand",
        color="color",
        color_discrete_map={"Selected Hour": "red", "Other Hours": "blue"},
        title=f"Predicted Bike Demand Throughout {date}",
        text="Predicted Demand"
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
