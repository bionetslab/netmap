import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


from torch.utils.data import DataLoader, TensorDataset


class ZINBLoss(nn.Module):
    def __init__(self, scale_factor=1.0, eps=1e-10, ridge_lambda=0.0):
        """
        Zero-Inflated Negative Binomial (ZINB) Loss
        Args:
            scale_factor (float): Scale factor applied to predictions.
            eps (float): Small value for numerical stability.
            ridge_lambda (float): Regularization weight for the zero-inflation probability (pi).
        """
        super(ZINBLoss, self).__init__()
        self.scale_factor = scale_factor
        self.eps = eps
        self.ridge_lambda = ridge_lambda

    def forward(self, y_true, y_pred, theta, pi):
        """
        Compute the ZINB loss.
        Args:
            y_true (torch.Tensor): Ground truth counts (non-negative integers).
            y_pred (torch.Tensor): Predicted mean values (mu).
            theta (torch.Tensor): Dispersion parameter (shape parameter).
            pi (torch.Tensor): Zero-inflation probability (between 0 and 1).
        Returns:
            torch.Tensor: ZINB negative log-likelihood.
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
    def __init__(self, input_dim, latent_dim, dropout_rate=0.0, hidden_dim = 128):
        super(ZINBAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after activation
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder for mean (mu)
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after activation
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()  # Ensure non-negative predictions
        )
        
        # Decoder for dispersion (theta)
        self.decoder_theta = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after activation
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()  # Ensure non-negative dispersion
        )
        
        # Decoder for zero-inflation probability (pi)
        self.decoder_pi = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after activation
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Ensure probability values between 0 and 1
        )
        
        self.zinb_loss = ZINBLoss()  # Use ZINBLoss for the computation
        self.forward_mu_only = False
        self.forward_theta_only = False
        self.latent_only = False
        self.forward_pi_only = False



    def forward(self, x):
        
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
        # Forward pass
        mu, theta, pi = self.forward(x)
        
        # Compute ZINB loss
        loss = self.zinb_loss(x, mu, theta, pi)
        return loss
    




    
def get_thetas(model, data_tensor):
    model.forward_mu_only = False
    model.forward_theta_only = True
    model.latent_only = False

    lat_mu = model(data_tensor.cuda())

    param = pd.DataFrame(lat_mu.detach().cpu().numpy())
    mean_theta = param.mean().values
    return mean_theta

def get_mus(model, data_tensor):
    model.forward_mu_only = True
    model.forward_theta_only = False
    model.latent_only = False

    lat_mu = model(data_tensor.cuda())

    param = pd.DataFrame(lat_mu.detach().cpu().numpy())
    mean_theta = param.mean().values
    return mean_theta


def get_mus_grouping(model, data_tensor, grouping):
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

