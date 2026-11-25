import pandas as pd
from typing import Dict, Any, Optional
from .config import HYPERPARAMETERS
from .data_loader import DataLoader
from .regime_analyzer import RegimeAnalyzer
from .autoencoder import AutoencoderTrainer
from .predictors import RFPredictor, LSTMPredictor

class PipelineOrchestrator:
    """
    Coordinates the execution of the macroeconomic analysis pipeline.
    Workflow: Load Data -> Feature Extraction (PCA/AE) -> Clustering -> Prediction.
    """

    def __init__(self, country_code: str = "US"):
        self.country_code = country_code
        self.loader = DataLoader(country_code)
        self.analyzer = RegimeAnalyzer(cluster_count=HYPERPARAMETERS["cluster_count"])
        self.ae_trainer = None # Initialized if requested

    def run_pipeline(self, target_df: pd.DataFrame, use_autoencoder: bool = False) -> Optional[Dict[str, Any]]:
        """
        Executes the full analysis pipeline.
        target_df: DataFrame containing the target variable (e.g. S&P 500 returns).
        Returns None if data loading fails.
        """
        try:
            # 1. Data Loading
            print(f"--- Starting Pipeline for {self.country_code} ---")
            macro_data = self.loader.load_data()
        except Exception as e:
            print(f"Critical error loading data for {self.country_code}: {e}")
            return None

        # Align macro data with target
        common_idx = macro_data.index.intersection(target_df.index)
        if len(common_idx) == 0:
            print(f"No overlapping dates found for {self.country_code}. Skipping.")
            return None

        macro_data = macro_data.loc[common_idx]
        target = target_df.loc[common_idx].squeeze()

        # 2. Feature Extraction
        if use_autoencoder:
            print("Using Autoencoder for feature extraction...")
            self.ae_trainer = AutoencoderTrainer(
                input_dim=macro_data.shape[1],
                latent_dim=HYPERPARAMETERS["autoencoder_latent_dim"]
            )
            self.ae_trainer.train(macro_data)
            features = self.ae_trainer.extract_features(macro_data)
        else:
            print("Using PCA for feature extraction...")
            features, var_explained = self.analyzer.perform_pca(
                macro_data,
                variance_threshold=HYPERPARAMETERS["variance_threshold"]
            )
            print(f"Explained Variance: {var_explained:.2%}")

        # 3. Regime Clustering
        print("Identifying market regimes...")
        labels = self.analyzer.perform_kmeans(features)

        # Combine features and labels for prediction
        final_features = features.join(labels)

        # 4. Prediction (Comparative Study)
        results = {}
        for pred_class in [RFPredictor, LSTMPredictor]:
            # Simplified split for demonstration
            split = int(len(final_features) * (1 - HYPERPARAMETERS["test_split"]))
            if split == 0:
                print(f"Insufficient data for training/testing split in {self.country_code}. Skipping predictor.")
                continue

            X_train, X_test = final_features.iloc[:split], final_features.iloc[split:]
            y_train, y_test = target.iloc[:split], target.iloc[split:]

            predictor = pred_class() if pred_class == RFPredictor else LSTMPredictor(
                input_dim=final_features.shape[1],
                hidden_dim=HYPERPARAMETERS["lstm_hidden_dim"],
                layers=HYPERPARAMETERS["lstm_layers"]
            )

            predictor.train(X_train, y_train)
            preds = predictor.predict(X_test)

            # Align y_test with predictions for windowed models (like LSTM)
            eval_y_test = y_test
            if isinstance(predictor, LSTMPredictor):
                # LSTM predicts from index 'window' onwards
                window = HYPERPARAMETERS["window_size"]
                eval_y_test = y_test.iloc[window:]

            metrics = predictor.evaluate(eval_y_test, preds)
            results[predictor.name] = metrics

        return {
            "metrics": results,
            "data": final_features,
            "target": target,
            "labels": labels
        }
