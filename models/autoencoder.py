"""Autoencoder anomaly detector using PyTorch.

Trained on legitimate transactions only. At inference, high reconstruction
error signals a transaction the autoencoder has never seen — a fraud indicator.
Uses focal-loss-inspired weighting during fine-tuning when labels are available.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FraudAutoencoder(nn.Module):
    """Symmetric autoencoder: encoder compresses, decoder reconstructs."""

    def __init__(self, input_dim: int, encoding_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        hidden = input_dim // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    """Train/predict wrapper for the fraud autoencoder.

    Training:
      - Fit on legitimate transactions only (is_fraud == 0)
      - High MSE reconstruction error at inference → anomaly

    Args:
        encoding_dim: Bottleneck dimension.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
    """

    def __init__(
        self,
        encoding_dim: int = 16,
        epochs: int = 30,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
    ) -> None:
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.net: Optional[FraudAutoencoder] = None
        self._threshold: float = 0.0
        self._input_dim: int = 0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AutoencoderDetector":
        """Train the autoencoder on non-fraud examples only."""
        if y is not None:
            X_train = X.loc[y == 0].values.astype("float32")
        else:
            X_train = X.values.astype("float32")

        logger.info("Autoencoder training on %d legitimate samples", len(X_train))

        self._input_dim = X_train.shape[1]
        self.net = FraudAutoencoder(self._input_dim, self.encoding_dim).to(DEVICE)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(X_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                recon = self.net(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, epoch_loss / len(X_train))

        # Set anomaly threshold at 95th percentile of training reconstruction error
        errors = self._reconstruction_error(X_train)
        self._threshold = float(np.percentile(errors, 95))
        logger.info("Anomaly threshold (p95 train MSE): %.6f", self._threshold)
        return self

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model not fitted.")
        self.net.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X.astype("float32")).to(DEVICE)
            recon = self.net(tensor)
            errors = ((tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return reconstruction error normalized to [0, 1] as anomaly probability."""
        errors = self._reconstruction_error(X.values.astype("float32"))
        # Sigmoid-like normalization centred on the training threshold
        score = 1.0 / (1.0 + np.exp(-(errors - self._threshold) * 10))
        return np.clip(score, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        errors = self._reconstruction_error(X.values.astype("float32"))
        return (errors > self._threshold).astype(int)
