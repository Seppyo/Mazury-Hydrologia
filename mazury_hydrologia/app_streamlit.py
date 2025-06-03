
import streamlit as st
from datetime import date, datetime, timedelta
import os
import pandas as pd

from fetch_sentinel import fetch_sentinel_images, fetch_weather_data
from analyzer import MazuryHydrologyAnalyzer

# --- KONFIGURACJA ---
DATA_DIR = "./data"
OUTPUT_DIR = "./output"

MAZURY_BBOX = [21.0, 53.3, 22.5, 54.0]  # [min_lon, min_lat, max_lon, max_lat]
MAZURY_CENTER_LAT = 53.65
MAZURY_CENTER_LON = 21.75

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.title("Analiza hydrologiczna Mazur")

start_date = st.date_input("Data startowa", date(2023, 1, 1))
end_date = st.date_input("Data końcowa", date.today())

if start_date > end_date:
    st.error("Data startowa musi być wcześniej niż data końcowa.")
    st.stop()

horizon = st.slider("Ile dni do przodu prognozować?", min_value=1, max_value=14, value=7)

if st.button("Pobierz dane i uruchom analizę"):
    # ---------- Pobieranie danych ----------
    def split_date_range(start_date_str, end_date_str, delta_days=30):
        ranges = []
        cur_start = datetime.strptime(start_date_str, "%Y-%m-%d")
        end = datetime.strptime(end_date_str, "%Y-%m-%d")
        while cur_start <= end:
            cur_end = min(cur_start + timedelta(days=delta_days-1), end)
            ranges.append((cur_start.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d")))
            cur_start = cur_end + timedelta(days=1)
        return ranges

    satellite_files = {}
    weather_csv = None

    date_ranges = split_date_range(str(start_date), str(end_date))

    with st.spinner("Pobieram obrazy Sentinel‑2..."):
        try:
            for dr_start, dr_end in date_ranges:
                files = fetch_sentinel_images(MAZURY_BBOX, dr_start, dr_end, DATA_DIR)
                satellite_files.update(files)
            st.success(f"Pobrano {len(satellite_files)} plików TIFF")
        except Exception as e:
            st.error(f"Błąd pobierania obrazów: {e}")
            st.stop()

    with st.spinner("Pobieram dane pogodowe..."):
        try:
            weather_csv = fetch_weather_data(
                MAZURY_CENTER_LAT,
                MAZURY_CENTER_LON,
                str(start_date),
                str(end_date),
                os.path.join(DATA_DIR, "weather.csv"),
            )
            st.success("Dane pogodowe pobrane")
        except Exception as e:
            st.error(f"Błąd pobierania pogody: {e}")
            st.stop()

    # ---------- Analiza ----------
    analyzer = MazuryHydrologyAnalyzer(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    changes_df = analyzer.run_complete_analysis(satellite_files, weather_csv)

    st.subheader("Zmiany powierzchni wód (km²)")
    st.line_chart(changes_df.set_index("date")["area_km2"])

    st.subheader("Prognozowana powierzchnia wód (km²)")
    forecasts_df = analyzer.forecast_water_area(changes_df, horizon_days=horizon)
    combo = pd.concat(
        [
            changes_df[["date", "area_km2"]].rename(columns={"area_km2": "area_km2"}),
            forecasts_df.rename(columns={"predicted_area_km2": "area_km2"}),
        ],
        ignore_index=True,
    ).sort_values("date")
    st.line_chart(combo.set_index("date"))

    st.subheader("Prognoza pogody (lokalna)")
    weather_future = analyzer.forecast_weather(horizon_days=horizon)
    st.line_chart(weather_future.set_index("date")[["temperature_2m_max", "precipitation_sum"]])

    st.subheader("Sezonowość powierzchni wód")
    if os.path.exists(os.path.join(OUTPUT_DIR, "seasonal_patterns.png")):
        st.image(os.path.join(OUTPUT_DIR, "seasonal_patterns.png"))

    st.subheader("Maski wody")
    for key in sorted(satellite_files.keys()):
        date_for_mask = key.split("_to_")[-1] if "_to_" in key else key
        mask_path = os.path.join(OUTPUT_DIR, f"water_mask_{date_for_mask}.png")
        if os.path.exists(mask_path):
            st.image(mask_path, caption=f"Maska – {date_for_mask}", use_container_width=True)

    st.subheader("Dane pogodowe (historyczne)")
    st.dataframe(pd.read_csv(weather_csv))
