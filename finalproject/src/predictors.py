import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from typing import Tuple, Dict

class BasePredictor(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates common regression metrics."""
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }

class RFPredictor(BasePredictor):
    """Random Forest Regression Predictor."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class LSTMPredictor(BasePredictor):
    """PyTorch LSTM Predictor for time-series forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, layers: int = 2, lr: float = 1e-3):
        super().__init__("LSTM")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.lr = lr

        # Define the LSTM architecture
        self.model = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, layers, batch_first=True),
            nn.Linear(hidden_dim, 1)
        )
        # Note: The above sequential is simplified. Actual LSTM output requires handling (output, (hn, cn))
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        out, (hn, cn) = self.lstm(x)
        # We only need the last time step's output
        out = self.fc(out[:, -1, :])
        return out

    def train(self, X: pd.DataFrame, y: pd.Series, epochs: int = 50, batch_size: int = 32):
        # Prepare temporal sequences (windows)
        X_seq, y_seq = self._create_sequences(X, y)

        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq).view(-1, 1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.model_train = True # custom flag
        for epoch in range(epochs):
            for batch in loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_seq, _ = self._create_sequences(X, pd.Series(np.zeros(len(X)), index=X.index)) # dummy y
        X_tensor = torch.FloatTensor(X_seq)

        with torch.no_grad():
            preds = self.forward(X_tensor)

        return preds.numpy().flatten()

    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Converts flat data into sliding window sequences for LSTM."""
        X_vals = X.values
        y_vals = y.values

        X_seq, y_seq = [], []
        for i in range(len(X_vals) - window):
            X_seq.append(X_vals[i:i+window])
            y_seq.append(y_vals[i+window])

        return np.array(X_seq), np.array(y_seq)

    def parameters(self):
        return list(self.lstm.parameters()) + list(self.fc.parameters())
