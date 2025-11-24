import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, date
import plotly.express as px

# =====================================================
# 1️⃣ Load Model & Scaler
# =====================================================
@st.cache_resource
def load_model():
    with open("xgb_hourly_demand_with_weather.pkl", "rb") as file:
        model_package = pickle.load(file)
    return model_package

model_package = load_model()
model = model_package["model"]
scaler = model_package["scaler"]
feature_columns = model_package["feature_columns"]

# =====================================================
# 2️⃣ Fetch Weather Data
# =====================================================
def fetch_weather(selected_date: date, latitude=51.5072, longitude=-0.1276):
    """Fetch weather for a given date (historical or forecast)."""
    today = datetime.utcnow().date()
    selected_str = selected_date.strftime("%Y-%m-%d")

    # Determine API type
    if selected_date < today:
        # Historical (ERA5 archive)
        url = (
            f"https://archive-api.open-meteo.com/v1/era5?"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={selected_str}&end_date={selected_str}"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloudcover"
        )
    else:
        # Forecast / current
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloudcover"
        )

    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        hourly = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            "temperature_2m": data["hourly"]["temperature_2m"],
            "relative_humidity_2m": data["hourly"].get("relative_humidity_2m", [np.nan]*24),
            "precipitation": data["hourly"].get("precipitation", [np.nan]*24),
            "wind_speed_10m": data["hourly"].get("wind_speed_10m", [np.nan]*24),
            "cloudcover": data["hourly"].get("cloudcover", [np.nan]*24)
        })
        return hourly
    except Exception as e:
        st.error(f"⚠️ Weather fetch failed: {e}")
        return None

# =====================================================
# 3️⃣ Feature Creation
# =====================================================
def create_features(date, hour, temp, humidity, precipitation, windspeed, cloudcover):
    dt = pd.to_datetime(date)
    season = ((dt.month % 12 + 3) // 3)

    row = {
        "hour_sin": np.sin(hour / 24 * 2 * np.pi),
        "hour_cos": np.cos(hour / 24 * 2 * np.pi),
        "month_sin": np.sin(dt.month / 12 * 2 * np.pi),
        "month_cos": np.cos(dt.month / 12 * 2 * np.pi),
        "season_sin": np.sin(season / 4 * 2 * np.pi),
        "season_cos": np.cos(season / 4 * 2 * np.pi),
        "dow_sin": np.sin(dt.weekday() / 7 * 2 * np.pi),
        "dow_cos": np.cos(dt.weekday() / 7 * 2 * np.pi),
        "is_weekend": int(dt.weekday() >= 5),
        "temperature_2m": temp,
        "relative_humidity_2m": humidity,
        "precipitation": precipitation,
        "wind_speed_10m": windspeed,
        "cloudcover": cloudcover
    }
    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

def predict_hourly_demand(input_df):
    scaled = scaler.transform(input_df)
    return model.predict(scaled)[0]

def predict_full_day(selected_date, temp, humidity, precipitation, windspeed, cloudcover):
    records = []
    for hour in range(24):
        df_hour = create_features(selected_date, hour, temp, humidity, precipitation, windspeed, cloudcover)
        pred = predict_hourly_demand(df_hour)
        records.append({"Hour": hour, "Predicted Demand": pred})
    return pd.DataFrame(records)

# =====================================================
# 4️⃣ Streamlit UI
# =====================================================
st.set_page_config(page_title="🚲 Bike Demand Forecast", layout="wide")

st.title("🚴 Predicting Hourly Bike Rental Demand with Weather Conditions")
st.markdown("Use real or forecasted weather data to predict hourly bike demand in London.")

col1, col2 = st.columns([2, 1])
with col1:
    selected_date = st.date_input("📅 Select Date", datetime(2025, 11, 24))
with col2:
    selected_hour = st.slider("⏰ Hour (0–23)", 0, 23, 17)

# Compact Weather Section
st.markdown("### 🌦 Weather Data (Auto or Manual)")
col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)

# Default values
temp, humidity, precip, windspeed, cloudcover = 10.0, 45.0, 0.0, 9.0, 40.0

# Fetch Weather Button
if st.button("🌤 Fetch Weather Data"):
    weather_df = fetch_weather(selected_date)
    if weather_df is not None:
        row = weather_df.iloc[selected_hour]
        temp = row["temperature_2m"]
        humidity = row["relative_humidity_2m"]
        precip = row["precipitation"]
        windspeed = row["wind_speed_10m"]
        cloudcover = row["cloudcover"]
        st.success(f"✅ Weather fetched for {selected_date} at {selected_hour}:00")

# Inputs (compact)
with col_w1:
    temp = st.number_input("Temp (°C)", value=float(temp))
with col_w2:
    humidity = st.number_input("Humidity (%)", value=float(humidity))
with col_w3:
    precip = st.number_input("Precip (mm)", value=float(precip))
with col_w4:
    windspeed = st.number_input("Wind (km/h)", value=float(windspeed))
with col_w5:
    cloudcover = st.number_input("Cloud (%)", value=float(cloudcover))

# =====================================================
# 5️⃣ Predict
# =====================================================
if st.button("🔮 Predict Demand"):
    df_input = create_features(selected_date, selected_hour, temp, humidity, precip, windspeed, cloudcover)
    pred = predict_hourly_demand(df_input)

    st.success(f"🚲 **Predicted Demand at {selected_hour}:00 → {pred:.0f} bikes/hour**")

    # Predict whole day
    daily_df = predict_full_day(selected_date, temp, humidity, precip, windspeed, cloudcover)
    daily_df["color"] = np.where(daily_df["Hour"] == selected_hour, "Selected Hour", "Other Hours")

    fig = px.bar(
        daily_df,
        x="Hour",
        y="Predicted Demand",
        color="color",
        color_discrete_map={"Selected Hour": "red", "Other Hours": "blue"},
        text="Predicted Demand",
        title=f"Predicted Bike Demand Throughout {selected_date}",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
