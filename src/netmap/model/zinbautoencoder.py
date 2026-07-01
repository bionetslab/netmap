"""Zero-Inflated Negative Binomial (ZINB) autoencoder for scRNA-seq count data.

This module implements a ZINB autoencoder with three decoder heads for the
mean (mu), dispersion (theta), and zero-inflation probability (pi) parameters.
Forward-mode flags (``forward_mu_only``, ``forward_theta_only``, ``latent_only``,
``forward_pi_only``) redirect :meth:`ZINBAutoencoder.forward` to return a single
tensor required by Captum attribution methods.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


from torch.utils.data import DataLoader, TensorDataset


class ZINBLoss(nn.Module):
    def __init__(self, scale_factor=1.0, eps=1e-10, ridge_lambda=0.0):
        """Zero-Inflated Negative Binomial loss module.

        Args:
            scale_factor (float): Scale factor applied to predictions. Defaults to 1.0.
            eps (float): Small value for numerical stability. Defaults to 1e-10.
            ridge_lambda (float): Ridge regularisation weight for the zero-inflation
                probability (pi). Defaults to 0.0.
        """
        super(ZINBLoss, self).__init__()
        self.scale_factor = scale_factor
        self.eps = eps
        self.ridge_lambda = ridge_lambda

    def forward(self, y_true, y_pred, theta, pi):
        """Compute the ZINB negative log-likelihood loss.

        Args:
            y_true (torch.Tensor): Ground truth counts (non-negative integers).
            y_pred (torch.Tensor): Predicted mean values (mu).
            theta (torch.Tensor): Dispersion parameter.
            pi (torch.Tensor): Zero-inflation probability in (0, 1).

        Returns:
            torch.Tensor: Mean ZINB negative log-likelihood, optionally with
                ridge penalty on pi.
        """
        eps = self.eps
        y_true = y_true.float()
        y_pred = y_pred.float() * self.scale_factor
        theta = theta.float()
        pi = torch.clamp(pi.float(), min=eps, max=1 - eps)  # Ensure pi is in (0, 1)

        # Clip theta to avoid numerical issues
        theta = torch.clamp(theta, max=1e6)

        # Negative binomial log-likelihood
        nb_case = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + eps)
            + (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps)))
            + y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )

        # Zero-inflation log-likelihood for y_true = 0
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        # Combine cases: zero or NB
        result = torch.where(y_true < eps, zero_case, nb_case)

        # Add ridge penalty for pi
        ridge = self.ridge_lambda * torch.square(pi)
        result += ridge

        return torch.mean(result)  # Return mean loss over the batch



class ZINBAutoencoder(nn.Module):
    """Autoencoder with three decoder heads for ZINB-distributed scRNA-seq data.

    The shared encoder compresses gene expression into a latent space. Three
    independent decoder branches predict the ZINB parameters:

    - **mu** (mean expression) via Softplus activation.
    - **theta** (dispersion) via Softplus activation.
    - **pi** (zero-inflation probability) via Softplus activation.

    Four mutually exclusive forward-mode flags redirect
    :meth:`forward` to return a single output tensor for Captum attribution:

    - ``forward_mu_only``: return only mu.
    - ``forward_theta_only``: return only theta.
    - ``latent_only``: return the latent embedding.
    - ``forward_pi_only``: return only pi.

    When all flags are ``False`` (default), ``forward`` returns ``(mu, theta, pi)``.

    Args:
        input_dim (int): Number of input genes.
        latent_dim (int): Dimensionality of the latent space.
        dropout_rate (float): Dropout probability after each hidden layer.
            Defaults to 0.0.
        hidden_dims (list of int): Width of each hidden layer in the encoder;
            all three decoders mirror this in reverse. Defaults to ``[64]``.
    """
    def __init__(self, input_dim, latent_dim, dropout_rate=0.0, hidden_dims = [64]):
        super(ZINBAutoencoder, self).__init__()

        # --- ENCODER ---
        encoder_layers = []

        # 1. Input layer (from input_dim to the first hidden layer)
        current_dim = input_dim

        # 2. Hidden layers (Loop through the specified dimensions)
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim # Update current dimension for the next layer

        # 3. Output layer (from the last hidden layer to latent_dim)
        # The input to this layer is the last element of hidden_dims
        final_hidden_dim = hidden_dims[-1] if hidden_dims else input_dim
        encoder_layers.append(nn.Linear(final_hidden_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        current_dim = latent_dim
        decoder_layers = []

        # 1. Hidden layers (Loop through reversed hidden dimensions)
        # Use reversed(hidden_dims) to go from latent_dim back up
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim # Update current dimension

        # 2. Output layer (from the first hidden layer back to input_dim)
        # The output of this layer is the original input_dim
        final_decode_dim = input_dim
        decoder_layers.extend([
            nn.Linear(current_dim, final_decode_dim),
            nn.Softplus() # Final activation
        ])

        self.decoder_mu = nn.Sequential(*decoder_layers)

        current_dim = latent_dim
        decoder_layers = []

        # 1. Hidden layers (Loop through reversed hidden dimensions)
        # Use reversed(hidden_dims) to go from latent_dim back up
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim # Update current dimension

        # 2. Output layer (from the first hidden layer back to input_dim)
        # The output of this layer is the original input_dim
        final_decode_dim = input_dim
        decoder_layers.extend([
            nn.Linear(current_dim, final_decode_dim),
            nn.Softplus() # Final activation
        ])

        self.decoder_theta = nn.Sequential(*decoder_layers)


        current_dim = latent_dim
        decoder_layers = []

        # 1. Hidden layers (Loop through reversed hidden dimensions)
        # Use reversed(hidden_dims) to go from latent_dim back up
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim # Update current dimension

        # 2. Output layer (from the first hidden layer back to input_dim)
        # The output of this layer is the original input_dim
        final_decode_dim = input_dim
        decoder_layers.extend([
            nn.Linear(current_dim, final_decode_dim),
            nn.Softplus() # Final activation
        ])

        self.decoder_pi = nn.Sequential(*decoder_layers)

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),  # Dropout after activation
        #     nn.Linear(hidden_dim, latent_dim)
        # )

        # # Decoder for mean (mu)
        # self.decoder_mu = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),  # Dropout after activation
        #     nn.Linear(hidden_dim, input_dim),
        #     nn.Softplus()  # Ensure non-negative predictions
        # )

        # # Decoder for dispersion (theta)
        # self.decoder_theta = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),  # Dropout after activation
        #     nn.Linear(hidden_dim, input_dim),
        #     nn.Softplus()  # Ensure non-negative dispersion
        # )

        # # Decoder for zero-inflation probability (pi)
        # self.decoder_pi = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),  # Dropout after activation
        #     nn.Linear(hidden_dim, input_dim),
        #     nn.Sigmoid()  # Ensure probability values between 0 and 1
        # )

        self.zinb_loss = ZINBLoss()  # Use ZINBLoss for the computation
        self.forward_mu_only = False
        self.forward_theta_only = False
        self.latent_only = False
        self.forward_pi_only = False



    def forward(self, x):
        """Run a forward pass, returning output controlled by the mode flags.

        When all mode flags are ``False``, returns ``(mu, theta, pi)``.
        Set one of ``forward_mu_only``, ``forward_theta_only``, ``latent_only``,
        or ``forward_pi_only`` to ``True`` before passing the model to a Captum
        explainer so that it returns a single tensor.

        Args:
            x (torch.Tensor): Input gene expression tensor of shape
                ``(n_cells, input_dim)``.

        Returns:
            torch.Tensor or tuple: Depending on the active flag —
                ``mu``, ``theta``, latent, ``pi``,
                or ``(mu, theta, pi)`` tuple.
        """
        latent = self.encoder(x)
        mu = self.decoder_mu(latent)
        theta = self.decoder_theta(latent)
        pi = self.decoder_pi(latent)

        #data = self.decoder_data(latent)
        if self.forward_theta_only:
            return theta
        elif self.forward_mu_only:
            return mu
        elif self.latent_only:
            return latent
        elif self.forward_pi_only:
            return pi
        else:
            return mu, theta, pi


    def compute_loss(self, x):
        """Compute the ZINB loss for a batch.

        Args:
            x (torch.Tensor): Input count matrix of shape ``(batch, input_dim)``.

        Returns:
            torch.Tensor: Scalar ZINB negative log-likelihood loss.
        """
        # Forward pass
        mu, theta, pi = self.forward(x)

        # Compute ZINB loss
        loss = self.zinb_loss(x, mu, theta, pi)
        return loss




def get_thetas(model, data_tensor):
    """Return the mean predicted dispersion (theta) across all cells.

    Args:
        model (ZINBAutoencoder): A trained ZINB autoencoder.
        data_tensor (torch.Tensor): Input data on CPU; moved to CUDA internally.

    Returns:
        numpy.ndarray: Mean theta value per gene, shape ``(input_dim,)``.
    """
    model.forward_mu_only = False
    model.forward_theta_only = True
    model.latent_only = False

    lat_mu = model(data_tensor.cuda())

    param = pd.DataFrame(lat_mu.detach().cpu().numpy())
    mean_theta = param.mean().values
    return mean_theta

def get_mus(model, data_tensor):
    """Return the mean predicted mean expression (mu) across all cells.

    Args:
        model (ZINBAutoencoder): A trained ZINB autoencoder.
        data_tensor (torch.Tensor): Input data on CPU; moved to CUDA internally.

    Returns:
        numpy.ndarray: Mean mu value per gene, shape ``(input_dim,)``.
    """
    model.forward_mu_only = True
    model.forward_theta_only = False
    model.latent_only = False

    lat_mu = model(data_tensor.cuda())

    param = pd.DataFrame(lat_mu.detach().cpu().numpy())
    mean_theta = param.mean().values
    return mean_theta


def get_mus_grouping(model, data_tensor, grouping):
    """Return the mean predicted mu per observation group.

    Args:
        model (ZINBAutoencoder): A trained ZINB autoencoder.
        data_tensor (torch.Tensor): Input data on CPU; moved to CUDA internally.
        grouping (array-like): Group label per cell, length ``n_cells``.

    Returns:
        dict: Mapping ``{group_label: numpy.ndarray}`` of mean mu per gene.
    """
    model.forward_mu_only = True
    model.forward_theta_only = False
    model.latent_only = False

    lat_mu = model(data_tensor.cuda())


    param = pd.DataFrame(lat_mu.detach().cpu().numpy())
    param['obs'] = np.array(grouping)

    param = param.groupby('obs').mean()
    dictionary_of_mus = {}
    for k in param.groupby('obs').mean().reset_index()['obs'].unique():
        dictionary_of_mus[k] = param[param.index==k].values.flatten()
    return dictionary_of_mus

def get_thetas_grouping(model, data_tensor, grouping):
    """Return the mean predicted theta per observation group.

    Args:
        model (ZINBAutoencoder): A trained ZINB autoencoder.
        data_tensor (torch.Tensor): Input data on CPU; moved to CUDA internally.
        grouping (array-like): Group label per cell, length ``n_cells``.

    Returns:
        dict: Mapping ``{group_label: numpy.ndarray}`` of mean theta per gene.
    """
    model.forward_mu_only = False
    model.forward_theta_only = True
    model.latent_only = False

    lat_mu = model(data_tensor.cuda())


    param = pd.DataFrame(lat_mu.detach().cpu().numpy())
    param['obs'] = np.array(grouping)

    param = param.groupby('obs').mean()
    dictionary_of_mus = {}
    for k in param.groupby('obs').mean().reset_index()['obs'].unique():
        dictionary_of_mus[k] = param[param.index==k].values.flatten()
    return dictionary_of_mus
