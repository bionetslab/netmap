import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


from torch.utils.data import DataLoader, TensorDataset


class NegativeBinomialLoss(nn.Module):
    def __init__(self, scale_factor=1.0, eps=1e-10):
        """
        Negative Binomial Loss
        Args:
            scale_factor (float): Scale factor applied to predictions.
            eps (float): Small value for numerical stability.
        """
        super(NegativeBinomialLoss, self).__init__()
        self.scale_factor = scale_factor
        self.eps = eps

    def forward(self, y_true, y_pred, theta):
        """
        Compute the Negative Binomial loss.
        Args:
            y_true (torch.Tensor): Ground truth counts (non-negative integers).
            y_pred (torch.Tensor): Predicted mean values (mu).
            theta (torch.Tensor): Dispersion parameter (shape parameter).
        Returns:
            torch.Tensor: Negative log-likelihood of the Negative Binomial distribution.
        """
        eps = self.eps
        y_true = y_true.float()
        y_pred = y_pred.float() * self.scale_factor
        theta = theta.float()

        # Clip theta to avoid numerical issues
        theta = torch.clamp(theta, max=1e6)

        # Negative binomial log-likelihood
        t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + \
             y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))

        loss = t1 + t2
        return torch.mean(loss)  # Return mean loss over the batch
    
class NegativeBinomialAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_rate=0.1, hidden_dims=[64], activation='relu'):
        super(NegativeBinomialAutoencoder, self).__init__()
        
        # Mapping for the requested activation functions
        acts = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU, 'sigmoid': nn.Sigmoid}
        activation_layer = acts.get(activation.lower(), nn.ReLU)

        # --- ENCODER ---
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, h_dim),
                activation_layer(), # Changed from nn.ReLU()
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim
            
        final_hidden_dim = hidden_dims[-1] if hidden_dims else input_dim
        encoder_layers.append(nn.Linear(final_hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
            
        # --- DECODER MU ---
        current_dim = latent_dim
        decoder_layers_mu = []
        for h_dim in reversed(hidden_dims):
            decoder_layers_mu.extend([
                nn.Linear(current_dim, h_dim),
                activation_layer(), # Changed from nn.ReLU()
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim

        decoder_layers_mu.extend([
            nn.Linear(current_dim, input_dim),
            nn.Softplus()
        ])
        self.decoder_mu = nn.Sequential(*decoder_layers_mu)

        # --- DECODER THETA ---
        current_dim = latent_dim
        decoder_layers_theta = []
        for h_dim in reversed(hidden_dims):
            decoder_layers_theta.extend([
                nn.Linear(current_dim, h_dim),
                activation_layer(), # Changed from nn.ReLU()
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim

        decoder_layers_theta.extend([
            nn.Linear(current_dim, input_dim),
            nn.Softplus()
        ])

        self.decoder_theta = nn.Sequential(*decoder_layers_theta)
    
        # Loss and flags
        self.nb_loss = NegativeBinomialLoss()
        self.mse_loss = nn.MSELoss()
        self.forward_mu_only = False
        self.forward_theta_only = False
        self.latent_only = False
        

    def forward(self, x):
        
        latent = self.encoder(x)
        mu = self.decoder_mu(latent)
        theta = self.decoder_theta(latent)
        #data = self.decoder_data(latent)
        if self.forward_theta_only:
            return theta
        elif self.forward_mu_only:
            return mu 
        elif self.latent_only:
            return latent
        else:
            return mu, theta

    
    def latent(self, x):
        return self.encoder(x)


    def compute_loss(self, x, l1_lambda=0.1):
        mu, theta = self.forward(x)
        nb_loss = self.nb_loss(x, mu, theta)
        return nb_loss
       



    
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

