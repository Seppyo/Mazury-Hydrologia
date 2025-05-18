import streamlit as st
from analyzer import MazuryHydrologyAnalyzer
import os
from datetime import datetime

st.set_page_config(page_title="Hydrologia Mazur – Analiza i Prognozy", layout="wide")

st.title("🌊 Hydrologia Mazur – analiza danych satelitarnych, pogodowych i predykcja ML")

# Inicjalizacja analizatora
analyzer = MazuryHydrologyAnalyzer()

# --- Sekcja 1: Pobieranie danych pogodowych ---
st.sidebar.header("☁️ Dane pogodowe")
lat = st.sidebar.number_input("Szerokość geograficzna", value=53.8)
lon = st.sidebar.number_input("Długość geograficzna", value=21.6)
start_date = st.sidebar.date_input("Data początkowa", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("Data końcowa", value=datetime(2020, 12, 31))

if st.sidebar.button("📥 Pobierz dane pogodowe"):
    try:
        weather_path = os.path.join(analyzer.data_dir, "weather.csv")
        analyzer.fetch_weather_data(lat, lon, str(start_date), str(end_date), weather_path)
        st.success("✅ Dane pogodowe zostały pobrane i zapisane.")
    except Exception as e:
        st.error(f"❌ Błąd: {e}")

# --- Sekcja 2: Wczytanie danych hydrologicznych ---
st.sidebar.header("🌊 Dane hydrologiczne")
hydro_file = st.sidebar.file_uploader("Plik CSV z danymi hydrologicznymi", type="csv")
hydro_path = None
if hydro_file:
    hydro_path = os.path.join(analyzer.data_dir, "hydro.csv")
    with open(hydro_path, "wb") as f:
        f.write(hydro_file.read())
    st.sidebar.success("✅ Załadowano dane hydrologiczne.")

# --- Sekcja 3: Wczytanie danych satelitarnych ---
st.sidebar.header("🛰 Dane satelitarne (NDWI)")
satellite_files = {}
for i in range(3):
    date_input = st.sidebar.date_input(f"Data obrazu {i+1}", key=f"date_{i}")
    tiff_file = st.sidebar.file_uploader(f"GeoTIFF dla {date_input}", type=["tif", "tiff"], key=f"tiff_{i}")
    if tiff_file:
        save_path = os.path.join(analyzer.data_dir, f"{date_input}.tif")
        with open(save_path, "wb") as f:
            f.write(tiff_file.read())
        satellite_files[str(date_input)] = save_path

# --- Sekcja 4: Uruchomienie analizy ---
if st.sidebar.button("🚀 Uruchom analizę"):
    if not hydro_path or not satellite_files:
        st.error("❌ Musisz dostarczyć dane hydrologiczne i przynajmniej jeden obraz satelitarny.")
    else:
        weather_path = os.path.join(analyzer.data_dir, "weather.csv")
        if not os.path.exists(weather_path):
            st.error("❌ Brakuje danych pogodowych. Pobierz je najpierw.")
        else:
            with st.spinner("🔍 Analiza w toku..."):
                analyzer.run_complete_analysis(
                    satellite_files=satellite_files,
                    hydro_file=hydro_path,
                    weather_file=weather_path
                )
            st.success("✅ Analiza zakończona!")

            # Wyświetl wyniki
            st.header("📈 Wyniki analizy")
            water_img = os.path.join(analyzer.output_dir, "water_changes.png")
            seasonal_img = os.path.join(analyzer.output_dir, "seasonal_patterns.png")

            if os.path.exists(water_img):
                st.image(water_img, caption="Zmiany powierzchni zbiorników wodnych")
            if os.path.exists(seasonal_img):
                st.image(seasonal_img, caption="Analiza sezonowości")

            st.subheader("🗺 Maski zbiorników wodnych")
            for date_str in satellite_files:
                mask_path = os.path.join(analyzer.output_dir, f"water_mask_{date_str}.png")
                if os.path.exists(mask_path):
                    st.image(mask_path, caption=f"Maska: {date_str}")
