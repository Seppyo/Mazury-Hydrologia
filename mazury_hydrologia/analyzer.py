import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from sklearn.linear_model import LinearRegression

class MazuryHydrologyAnalyzer:
    def __init__(self, data_dir='./data', output_dir='./output'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.satellite_data = None
        self.weather_data = None
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def load_satellite_data(self, file_path):
        with rasterio.open(file_path) as src:
            self.satellite_data = {
                'data': src.read(),
                'meta': src.meta,
                'bounds': src.bounds,
                'crs': src.crs
            }

    def load_weather_data(self, file_path):
        self.weather_data = pd.read_csv(file_path, parse_dates=['date'])

    def calculate_ndwi(self):
        nir = self.satellite_data['data'][3].astype(float)  # NIR (B08)
        green = self.satellite_data['data'][1].astype(float)  # Green (B03)
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
            area_km2 = area_pixels * 100 / 1_000_000  # 10m resolution -> 100 m² per pixel
            records.append({'date': date, 'area_km2': area_km2})
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.output_dir, "water_changes.csv"), index=False)
        return df

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
        plt.axis('off')
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
        plt.title("Sezonowość powierzchni wód")
        plt.xlabel("Miesiąc")
        plt.ylabel("Powierzchnia (km²)")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "seasonal_patterns.png")
        plt.savefig(path)
        plt.close()

    def run_complete_analysis(self, satellite_files, weather_file):
        self.load_weather_data(weather_file)

        time_series = []
        for date_key, path in satellite_files.items():
            if "_to_" in date_key:
                start_str, end_str = date_key.split("_to_")
                date_str = end_str
            else:
                date_str = date_key
            self.load_satellite_data(path)
            ndwi = self.calculate_ndwi()
            mask = self.extract_water_bodies(ndwi)
            self.visualize_water_mask(mask, date_str)
            time_series.append((datetime.strptime(date_str, "%Y-%m-%d"), mask))

        changes_df = self.calculate_water_area_changes(time_series)
        self.visualize_water_changes(changes_df)
        self.analyze_seasonal_patterns(changes_df)
        return changes_df

    def forecast_weather(self, horizon_days=30):
        if self.weather_data is None:
            raise ValueError("Weather data not loaded.")

        df = self.weather_data.copy()

        df = df.dropna(subset=["temperature_2m_max", "precipitation_sum"])

        df["day_number"] = (df["date"] - df["date"].min()).dt.days

        forecasts = {}
        for feat in ["temperature_2m_max", "precipitation_sum"]:
            m = LinearRegression()

            valid_idx = df[feat].notna()
            X = df.loc[valid_idx, ["day_number"]]
            y = df.loc[valid_idx, feat]

            m.fit(X, y)

            future_days = pd.DataFrame({
                "day_number": np.arange(df["day_number"].max() + 1, df["day_number"].max() + 1 + horizon_days)
            })

            pred = m.predict(future_days)

            future_dates = pd.date_range(df["date"].max() + timedelta(days=1), periods=horizon_days)
            forecasts[feat] = pd.DataFrame({
                "date": future_dates,
                feat: pred
            })

        df_forecast = forecasts["temperature_2m_max"].merge(
            forecasts["precipitation_sum"], on="date", how="left"
        )
        return df_forecast

    def forecast_water_area(self, changes_df, horizon_days=30):
        if changes_df is None or changes_df.empty:
            raise ValueError("Water changes data not provided or empty.")

        df = changes_df.copy()
        df = df.dropna(subset=["area_km2"])
        df["day_number"] = (df["date"] - df["date"].min()).dt.days

        m = LinearRegression()
        X = df[["day_number"]]
        y = df["area_km2"]
        m.fit(X, y)

        future_days = pd.DataFrame({
            "day_number": np.arange(df["day_number"].max() + 1, df["day_number"].max() + 1 + horizon_days)
        })

        pred = m.predict(future_days)
        future_dates = pd.date_range(df["date"].max() + timedelta(days=1), periods=horizon_days)

        forecast_df = pd.DataFrame({
            "date": future_dates,
            "area_km2_forecast": pred
        })

        return forecast_df
