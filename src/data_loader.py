import requests
import pandas as pd
import os
from typing import Dict, Optional, List
from .config import FRED_API_KEY, COUNTRY_PROFILES

class DataLoader:
    """
    Handles acquisition of macroeconomic data from the FRED API
    and local CSV files, ensuring alignment across different frequencies.
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, country_code: str = "US"):
        if country_code not in COUNTRY_PROFILES:
            raise ValueError(f"Country code {country_code} not supported. Choose from {list(COUNTRY_PROFILES.keys())}")

        self.country_code = country_code
        self.profile = COUNTRY_PROFILES[country_code]

    def fetch_fred_series(self, series_id: str, name: str, start_date: str = "2000-01-01") -> Optional[pd.DataFrame]:
        """Fetches a single time series from FRED."""
        if not FRED_API_KEY:
            print(f"Error: FRED_API_KEY not found. Cannot fetch {name}.")
            return None

        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start_date,
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            if response.status_code != 200:
                print(f"Error fetching {name} ({series_id}): {response.status_code}")
                return None

            data = response.json()
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['date', 'value']].rename(columns={'value': name}).set_index('date')
        except Exception as e:
            print(f"Exception while fetching {name}: {e}")
            return None

    def load_data(self, use_api: bool = True, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Orchestrates data loading.
        If use_api is True, fetches based on country profile.
        If use_api is False, loads from provided csv_path.
        """
        if use_api:
            return self._load_from_api()
        elif csv_path and os.path.exists(csv_path):
            return pd.read_csv(csv_path, index_col='date', parse_dates=True)
        else:
            raise FileNotFoundError(f"No API access and CSV path {csv_path} is invalid.")

    def _load_from_api(self) -> pd.DataFrame:
        """Internal method to fetch all variables for the configured country."""
        all_series = []

        # Fetch Macro Variables
        for name, sid in self.profile["variables"].items():
            series_df = self.fetch_fred_series(sid, name)
            if series_df is not None:
                all_series.append(series_df)

        # Fetch Target Variable (if applicable/available)
        target_name = self.profile["target"]
        # Note: In a real scenario, target IDs would also be in COUNTRY_PROFILES
        # For now, we'll assume targets can be fetched if they have a valid FRED ID
        # or we might need a separate map. Let's assume targets are part of the API fetch
        # logic if we had their IDs.

        if not all_series:
            raise RuntimeError("Failed to fetch any data from FRED API.")

        # Align and clean
        final_df = pd.concat(all_series, axis=1).sort_index()
        final_df = final_df.ffill().bfill()

        return final_df

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Saves dataframe to the project data directory."""
        os.makedirs('finalproject/data', exist_ok=True)
        output_path = os.path.join('finalproject/data', filename)
        df.to_csv(output_path)
        print(f"Data saved to {output_path}")
