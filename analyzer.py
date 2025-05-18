import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

import mysql.connector
from config import MYSQL_CONFIG


class MazuryHydrologyAnalyzer:
    def __init__(self, data_dir='./data', output_dir='./output'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.satellite_data = None
        self.hydrological_data = None
        self.weather_data = None
        self.model = None
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def fetch_weather_data(self, lat, lon, start_date, end_date, save_path):
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            "&daily=temperature_2m_max,precipitation_sum"
            "&timezone=Europe%2FWarsaw"
        )
        r = requests.get(url)
        if r.status_code == 200:
            df = pd.DataFrame(r.json()['daily'])
            df['date'] = pd.to_datetime(df['time'])
            df.to_csv(save_path, index=False)
        else:
            raise Exception("Nie można pobrać danych pogodowych")

    def load_satellite_data(self, file_path):
        with rasterio.open(file_path) as src:
            self.satellite_data = {
                'data': src.read(),
                'meta': src.meta,
                'bounds': src.bounds,
                'crs': src.crs
            }

    def load_hydrological_data(self, file_path):
        self.hydrological_data = pd.read_csv(file_path, parse_dates=['date'])

    def load_weather_data(self, file_path):
        self.weather_data = pd.read_csv(file_path, parse_dates=['date'])

    def calculate_ndwi(self):
        nir = self.satellite_data['data'][3].astype(float)
        green = self.satellite_data['data'][2].astype(float)
        denom = nir + green
        ndwi = np.zeros_like(denom)
        valid = denom != 0
        ndwi[valid] = (green[valid] - nir[valid]) / denom[valid]
        return ndwi

    def extract_water_bodies(self, ndwi, threshold=0.3):
        return ndwi > threshold

    def calculate_water_area_changes(self, time_series_data):
        records = []
        for date, mask in time_series_data:
            area_pixels = np.sum(mask)
            area_km2 = area_pixels * 100 / 1_000_000  # 10m resolution
            records.append({'date': date, 'area_km2': area_km2})
        return pd.DataFrame(records)

    def visualize_water_changes(self, df):
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['area_km2'], marker='o')
        plt.title("Zmiany powierzchni wody")
        plt.xlabel("Data")
        plt.ylabel("Powierzchnia (km²)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "water_changes.png")
        plt.savefig(path)
        plt.close()

    def visualize_water_mask(self, mask, date_str):
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap='Blues')
        plt.title(f"Maska wody – {date_str}")
        path = os.path.join(self.output_dir, f"water_mask_{date_str}.png")
        plt.savefig(path)
        plt.close()

    def analyze_seasonal_patterns(self, df):
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].apply(
            lambda m: 'Zima' if m in [12, 1, 2] else
                      'Wiosna' if m in [3, 4, 5] else
                      'Lato' if m in [6, 7, 8] else 'Jesień'
        )
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='month', y='area_km2', data=df)
        plt.title("Sezonowość")
        plt.xlabel("Miesiąc")
        plt.ylabel("Powierzchnia (km²)")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "seasonal_patterns.png")
        plt.savefig(path)
        plt.close()

    def train_prediction_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_r2 = r2_score(y_test, rf.predict(X_test))

        xgb_model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_r2 = r2_score(y_test, xgb_model.predict(X_test))

        self.model = xgb_model if xgb_r2 > rf_r2 else rf

    def predict_future_changes(self, future_X):
        if self.model:
            return self.model.predict(future_X)
        return None

    def save_to_mysql(self, df):
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO wyniki (data, powierzchnia_km2, temperatura_max, opady, prognoza_km2)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                row['date'],
                row['area_km2'],
                row.get('temperature_2m_max'),
                row.get('precipitation_sum'),
                row.get('prediction')
            ))
        conn.commit()
        cursor.close()
        conn.close()

    def run_complete_analysis(self, satellite_files, hydro_file, weather_file):
        self.load_hydrological_data(hydro_file)
        self.load_weather_data(weather_file)

        time_series = []
        for date_str, path in satellite_files.items():
            self.load_satellite_data(path)
            ndwi = self.calculate_ndwi()
            mask = self.extract_water_bodies(ndwi)
            self.visualize_water_mask(mask, date_str)
            time_series.append((datetime.strptime(date_str, "%Y-%m-%d"), mask))

        changes_df = self.calculate_water_area_changes(time_series)
        self.visualize_water_changes(changes_df)
        self.analyze_seasonal_patterns(changes_df)

        merged = pd.merge(changes_df, self.weather_data, on='date', how='left')
        merged.dropna(inplace=True)

        if not merged.empty:
            X = merged[['temperature_2m_max', 'precipitation_sum']]
            y = merged['area_km2']
            self.train_prediction_model(X, y)

            future_preds = self.predict_future_changes(X)
            merged['prediction'] = future_preds
            self.save_to_mysql(merged)
