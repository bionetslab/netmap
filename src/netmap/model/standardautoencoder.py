import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


from torch.utils.data import DataLoader, TensorDataset

class MSELoss(nn.Module):
    def __init__(self, scale_factor=1.0):
        """
        Mean Squared Error Loss
        Args:
            scale_factor (float): Scale factor applied to predictions.
        """
        super(MSELoss, self).__init__()
        self.scale_factor = scale_factor
        self.mse_loss = nn.MSELoss()

    def forward(self, y_true, y_pred):
        """
        Compute the MSE loss.
        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted mean values.
        Returns:
            torch.Tensor: Mean Squared Error.
        """
        y_true = y_true.float()
        y_pred = y_pred.float() * self.scale_factor
        return self.mse_loss(y_true, y_pred)


class LogCoshLoss(nn.Module):
    def __init__(self):
        """
        Log-Cosh Loss
        """
        super(LogCoshLoss, self).__init__()

    def forward(self, y_true, y_pred):
        """
        Compute the Log-Cosh loss.
        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.
        Returns:
            torch.Tensor: Log-Cosh loss.
        """
        y_true = y_true.float()
        y_pred = y_pred.float()
        diff = y_pred - y_true
        loss = torch.log(torch.cosh(diff))
        return torch.mean(loss)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_rate=0.0, hidden_dim=128, loss_fn='mse'):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()  # Changed Softplus to ReLU based on typical usage for these losses
        )

        self.latent_only = False
        self.loss_fn_name = loss_fn

        if loss_fn == 'mse':
            self.loss_fn = MSELoss()
        elif loss_fn == 'logcosh':
            self.loss_fn = LogCoshLoss()
        else:
            raise ValueError("Loss function must be 'mse' or 'logcosh'.")

    def forward(self, x):
        latent = self.encoder(x)
        data = self.decoder(latent)
        if self.latent_only:
            return latent
        else:
            return data

    def latent(self, x):
        return self.encoder(x)

    def compute_loss(self, x, l1_lambda=0.1):
        reconstructed_x = self.forward(x)
        loss = self.loss_fn(x, reconstructed_x)
        return loss