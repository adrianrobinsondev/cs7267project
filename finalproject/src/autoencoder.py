import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from typing import Tuple

class Autoencoder(nn.Module):
    """
    PyTorch-based Autoencoder for non-linear dimensionality reduction.
    Individual contribution for the generalizable framework.
    """
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class AutoencoderTrainer:
    """
    Wrapper for training and using the Autoencoder model.
    """
    def __init__(self, input_dim: int, latent_dim: int = 4, lr: float = 1e-3):
        self.model = Autoencoder(input_dim, latent_dim)
        self.lr = lr
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def train(self, X: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> float:
        """
        Trains the autoencoder on the provided dataframe.
        Returns the final Mean Squared Error (MSE).
        """
        # Normalize data (standard scaling assumed to be done outside or we do it here)
        # For simplicity, we assume X is already scaled.
        X_tensor = torch.FloatTensor(X.values)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                inputs = batch[0]

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Optional: print progress every 10 epochs
            # if (epoch + 1) % 10 == 0:
            #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

        # Final loss on whole dataset
        self.model.eval()
        with torch.no_grad():
            final_outputs = self.model(X_tensor)
            final_loss = criterion(final_outputs, X_tensor).item()

        return final_loss

    def extract_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input data into its latent representation.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            latent = self.model.encode(X_tensor)

        column_names = [f"AE_{i+1}" for i in range(self.latent_dim)]
        return pd.DataFrame(latent.numpy(), index=X.index, columns=column_names)
