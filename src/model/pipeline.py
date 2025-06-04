import sys
sys.path.append('/data_nfs/og86asub/netmap/netmap-evaluation/netmap')

import scanpy as sc
import time 

from netmap.src.utils.misc import write_config

from netmap.src.model.negbinautoencoder import *
import scanpy as sc

from sklearn.model_selection import train_test_split
import time
from captum.attr import GradientShap
from netmap.src.model.inferrence_simple import *
from netmap.src.utils.data_utils import attribution_to_anndata
from netmap.src.model.pipeline import *
import numpy as np


from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN

import os.path as op
import os

def create_model_zoo(data_tensor, n_models = 4, n_epochs = 500):
    model_zoo = []
    for _ in range(n_models):

        data_train2, data_test2 = train_test_split(data_tensor,test_size=0.0, shuffle=True)

        trained_model2 = NegativeBinomialAutoencoder(input_dim=data_tensor.shape[1], latent_dim=10, dropout_rate = 0.25)
        trained_model2 = trained_model2.cuda()

        optimizer2 = torch.optim.Adam(trained_model2.parameters(), lr=1e-4)

        trained_model2 = train_autoencoder(
                trained_model2,
                data_train2.cuda(),
                optimizer2,
                num_epochs=n_epochs

            )
        model_zoo.append(trained_model2)
    return model_zoo


from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN


def set_latent_true(model_zoo):
    for mo in model_zoo:
        mo.forward_mu_only = False
        mo.forward_theta_only = False
        mo.latent_only = True
    return model_zoo


def set_all_false(model_zoo):
    for mo in model_zoo:
        mo.forward_mu_only = False
        mo.forward_theta_only = False
        mo.latent_only = False
    return model_zoo


def pair_cooccurence(u, v):
  u_arr = np.array(u)
  v_arr = np.array(v)
  term1 = np.sum(u_arr-v_arr!=0)
  return term1 / len(u_arr)

def compute_consensus_clustering(model_zoo, data_tensor, adata):
    model_zoo = set_latent_true(model_zoo)
    clusterings = {}
    for m in range(len(model_zoo)):
        latent = model_zoo[m](data_tensor.cuda())
        latent = sc.AnnData(latent.detach().cpu().numpy())
        sc.pp.neighbors(latent)
        sc.tl.umap(latent)
        latent.obs = adata.obs
        sc.tl.leiden(latent, resolution=0.1)
        clusterings[f'leiden_{m}'] = latent.obs[f'leiden']
    model_zoo = set_all_false(model_zoo)

    clusterings = pd.DataFrame(clusterings)

    latent.obs = pd.concat([latent.obs, clusterings], axis=1)
    clustering = clusterings.values.astype(int) 

    pairwise_distances_flat = pdist(clustering, metric=pair_cooccurence)
    pairwise_distances_matrix = squareform(pairwise_distances_flat)

    hdb = HDBSCAN(min_cluster_size=50, metric= 'precomputed')
    hdb.fit(pairwise_distances_matrix)
    latent.obs['consensus'] = hdb.labels_


    return latent

def set_mu_true(model_zoo):
    # forward only mu
    for mo in model_zoo:
        mo.forward_mu_only = True
        mo.forward_theta_only = False
        mo.latent_only = False
    return model_zoo

import numpy as np
from scipy.stats import nbinom

def draw_neg_binomial(mu, theta, size=1):
    """
    Draws random samples from a negative binomial distribution given mu and theta.

    Args:
        mu (float): The mean of the negative binomial distribution.
        theta (float): The dispersion parameter (often denoted as 'alpha' in some contexts,
                    related to the number of successes).
        size (int or tuple of ints, optional): The number of samples to draw.
                                            Default is 1.

    Returns:
        ndarray or scalar: Drawn samples from the negative binomial distribution.
    """
    if mu <= 0 or theta <= 0:
        raise ValueError("mu and theta must be positive.")

    p = theta / (mu + theta)
    n = theta  # theta in this parameterization corresponds to 'n' in scipy

    # scipy.stats.nbinom uses 'n' (number of successes) and 'p' (probability of success)
    samples = nbinom.rvs(n, p, size=size)
    return samples

def sample_from_vectors(mus, thetas, size=1):
    """
    Draws random samples from negative binomial distributions defined by
    corresponding elements in the mu and theta vectors.

    Args:
        mus (np.ndarray): A vector of mean parameters.
        thetas (np.ndarray): A vector of dispersion parameters.
        size (int or tuple of ints, optional): The number of samples to draw
                                            for each mu-theta pair. Default is 1.

    Returns:
        np.ndarray: An array of samples. If size is 1, the output will have the
                    same shape as mus and thetas. If size is greater than 1,
                    the output will have an additional dimension for the samples.
    """
    mus = np.asarray(mus)
    thetas = np.asarray(thetas)

    if mus.shape != thetas.shape:
        raise ValueError("mus and thetas vectors must have the same shape.")

    num_distributions = mus.shape[0]
    all_samples = []

    for i in range(num_distributions):
        mu_i = mus[i]
        theta_i = thetas[i]
        samples_i = draw_neg_binomial(mu_i, theta_i, size=size)
        all_samples.append(samples_i)

    if size == 1:
        return np.array(all_samples).flatten()
    else:
        return np.array(all_samples)



def generate_background_data(model_zoo, data_tensor, latent):
    backgrounds = {}
    for i in latent.obs['consensus'].unique():
        if i!=-1:
            backgrounds[i] = []

    for m in range(len(model_zoo)):
        # Get average params over all 
        mean_theta = get_thetas(model_zoo[m], data_tensor)
        mean_mu = get_mus(model_zoo[m], data_tensor)
        samples_mean = sample_from_vectors(mean_mu, mean_theta, size=200).T


        mean_mus = get_mus_grouping(model_zoo[m], data_tensor, latent.obs['consensus'])
        mean_thetas = get_thetas_grouping(model_zoo[m], data_tensor, latent.obs['consensus'])

        for i in backgrounds.keys():
            backgrounds[i].append(sample_from_vectors(mean_mus[i], mean_thetas[i], size=200).T)

    for b in backgrounds.keys():
        backgrounds[b] = np.concatenate(backgrounds[b])
        backgrounds[b] = torch.tensor(backgrounds[b]).cuda()

    return backgrounds


def compute_attributions(model_zoo, data_tensor, latent, gene_names, backgrounds, adata, n_top = 250):
    myexplainers = [GradientShap(mo) for mo in model_zoo]
    attributions = []
    for b in backgrounds.keys():
        aggregated_attribution, cou = inference_complete(myexplainers, data_tensor[latent.obs['consensus'] == b].cuda(), gene_names, backgrounds[b], xai_type='shap-like', num_iterations=5,n_top_genes=n_top )
        ad = attribution_to_anndata(aggregated_attribution, var=cou, obs=latent.obs[latent.obs['consensus'] == b])
        attributions.append(ad)
    return attributions
    
def concatenate_cluster_anndatas(attributions):
    aa = sc.concat(attributions, join = 'outer')
    # Concatenation somehow deletes the content of var
    aa.var['source']   = [l[0] for l in aa.var.index.str.split('_', expand=True)]
    aa.var['target']   = [l[1] for l in aa.var.index.str.split('_', expand=True)]
    return aa

import scipy.sparse as scs

def run_netmap(config, dataset_config):

    start_total = time.monotonic()
    
    ## Load config and setup outputs
    os.makedirs(config.output_directory, exist_ok=True)
    sc.settings.figdir = config.output_directory
    config.write_yaml(yaml_file=op.join(config.output_directory, 'config.yaml'))

    ## load data
    adata = sc.read_h5ad(config.input_data)
    

    ## Get the data matrix from the CustumAnndata obeject
    gene_names = np.array(adata.var.index)
    model_start = time.monotonic()

    if config.layer == 'counts':
        data_tensor = adata.layers['counts']
    else:
        data_tensor = adata.X

    if scs.issparse(data_tensor):
        data_tensor = torch.tensor(data_tensor.todense(), dtype=torch.float32)
    else:
        data_tensor = torch.tensor(data_tensor.X, dtype=torch.float32)


    print(data_tensor.shape)

    model_zoo = create_model_zoo(data_tensor, n_models=config.n_models, n_epochs=1500)

    latent = compute_consensus_clustering(model_zoo, data_tensor, adata)
    backgrounds = generate_background_data(model_zoo, data_tensor, latent)
    attributions = compute_attributions(model_zoo, data_tensor, latent, gene_names, backgrounds, adata, config.n_top_edges)
    grn_adata = concatenate_cluster_anndatas(attributions)
    model_elapsed = time.monotonic()-model_start
    grn_adata.write_h5ad(op.join(config.output_directory,config.adata_filename))

    time_elapsed_total = time.monotonic()-start_total


    res = {'time_elapsed_total': time_elapsed_total, 'time_elapsed_netmap': model_elapsed} 
    write_config(res, file=op.join(config.output_directory, 'results.yaml'))
  
