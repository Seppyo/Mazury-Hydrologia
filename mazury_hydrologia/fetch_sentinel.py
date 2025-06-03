import pandas as pd
import requests
import os

# --- KONFIGURACJA ---
SENTINEL_HUB_CLIENT_ID = "c68ae833-d042-4de7-a82d-c80a2e0dbb7c"
SENTINEL_HUB_CLIENT_SECRET = "lZk5EOyzvBCGboqijHnmCDxobr1mcfuu"
SENTINEL_HUB_API_URL = "https://services.sentinel-hub.com/api/v1/process"

def get_access_token(client_id: str, client_secret: str) -> str:
    url = "https://services.sentinel-hub.com/oauth/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

def build_payload(bbox, start_date, end_date):
    """Przygotowuje zapytanie do Sentinel Hub Process API."""
    return {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                "bbox": bbox,
            },
            "data": [
                {
                    "type": "S2L2A",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date}T00:00:00Z",
                            "to": f"{end_date}T23:59:59Z"
                        }
                    }
                }
            ]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }
            ]
        },
        "evalscript": """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02", "B08"],
                    output: { bands: 4 }
                };
            }
            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02, sample.B08];
            }
        """
    }

def fetch_sentinel_images(bbox, start_date, end_date, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    token = get_access_token(SENTINEL_HUB_CLIENT_ID, SENTINEL_HUB_CLIENT_SECRET)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = build_payload(bbox, start_date, end_date)

    response = requests.post(SENTINEL_HUB_API_URL, json=payload, headers=headers)
    response.raise_for_status()

    filename = os.path.join(save_dir, f"sentinel_{start_date}_to_{end_date}.tiff")
    with open(filename, "wb") as f:
        f.write(response.content)

    return {f"{start_date}_to_{end_date}": filename}

def fetch_weather_data(lat, lon, start_date, end_date, save_path):
    """Pobiera dane pogodowe z Open‑Meteo i zapisuje do CSV."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_max,precipitation_sum"
        "&timezone=Europe%2FWarsaw"
    )
    r = requests.get(url)
    r.raise_for_status()
    df = pd.DataFrame(r.json()['daily'])
    df['date'] = pd.to_datetime(df['time'])
    df.to_csv(save_path, index=False)
    return save_path
