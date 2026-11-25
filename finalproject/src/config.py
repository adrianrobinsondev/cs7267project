import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Country Profiles: Mapping of countries to FRED series IDs
# These should be expanded based on available FRED data for each region
COUNTRY_PROFILES = {
    "US": {
        "variables": {
            "CPI": "CPIAUCS", # Consumer Price Index for All Urban Consumers
            "FEDFUNDS": "FEDFUNDS", # Effective Federal Funds Rate
            "UNRATE": "UNRATE", # Unemployment Rate
            "10Y_YIELD": "GS10", # 10-Year Treasury Constant Maturity Rate
        },
        "target": "SP500", # Target variable (e.g., S&P 500)
    },
    "Japan": {
        "variables": {
            "CPI": "JPNCPIALLMIN", # Japan Consumer Price Index
            "INTEREST_RATE": "INTDSRJPM193N", # Interest rate (Example)
            "UNRATE": "JPUNRATE", # Japan Unemployment Rate
            "10Y_YIELD": "IRLTLT01JPM193N", # 10-Year Government Bond Yield
        },
        "target": "NIKKEI225",
    },
    "Korea": {
        "variables": {
            "CPI": "KORCPIALLMIN", # Korea Consumer Price Index
            "INTEREST_RATE": "KORINT", # Example Interest Rate
            "UNRATE": "KORUNRATE", # Korea Unemployment Rate
            "10Y_YIELD": "KOR10Y", # Example 10-Year Yield
        },
        "target": "KOSPI",
    },
}

# Global Hyperparameters
HYPERPARAMETERS = {
    "window_size": 5,            # Prediction window (e.g., 5-day returns)
    "variance_threshold": 0.95,   # PCA Explained Variance threshold
    "cluster_count": 3,           # Default number of market regimes (k)
    "lstm_hidden_dim": 64,        # LSTM hidden layer size
    "lstm_layers": 2,             # Number of LSTM layers
    "autoencoder_latent_dim": 4,  # Latent dimension for Autoencoder
    "test_split": 0.2,            # Percentage of data for testing
}
