import sys
  
# append the path of the parent directory
sys.path.append("..")

import captum
from datetime import datetime
import os
import shutil
import pandas as pd
import numpy as np
import scanpy as sc
import time
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import LRP
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os.path as op


from utils.tf_utils import *
import models.model as m
from models.model import train_multi_task_model

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
    def __init__(self, input_dim, latent_dim, dropout_rate=0.0, hidden_dim = 128):
        super(NegativeBinomialAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Dropout after activation
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()  
        )
        
        self.decoder_theta = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after activation
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()  # Ensure non-negative dispersion
        )
        
        self.decoder_data = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after activation
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        
        self.nb_loss =  NegativeBinomialLoss()
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
       


def train_autoencoder(
    model,
    data_train,
    optimizer,
    batch_size=32,  # Minibatch size
    num_epochs=100,
):
    # Prepare DataLoader for training
    train_dataset = TensorDataset(data_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # Track loss for the epoch

        for batch in train_loader:
            data_batch = batch[0]  # Unpack the single-element tuple from TensorDataset
            # Forward pass
            optimizer.zero_grad()
            loss = model.compute_loss(data_batch)

            # Compute total loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate loss for the epoch

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    return model


    
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

