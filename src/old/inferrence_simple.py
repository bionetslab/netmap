
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import hstack

import numpy as np
from scipy.stats import nbinom

def inference_complete(lrp, data, gene_names, background, xai_type= 'lrp-like', num_iterations=10, n_top_genes=500) -> pd.DataFrame:
    """
    Compute the LRP values for all targets (All genes in anndata object)
    Filter attribution values

    Args:
    configuration_dict: configuration of trained models
    data: Passed as CustomAnndataLoader instance
    tf_gene_names: List of genes to be considered as transcription factors.

    returns: List of target wise attributions per edge
    """
    
    name_list = []
    ## Presumably all genes in adata var have been used as target genes 
    ### This is currentlyWednesday unlikely due to random sampling
    top_values = None
    attribution_list =  []
    target_names = []
    for g in tqdm(range(data.shape[1])):
        attribution_all, names = inference_one_target(g, lrp, data, gene_names, background, 
                                                        xai_type=xai_type, n_top_genes = n_top_genes, num_iterations=num_iterations)
        attribution_list.append(attribution_all)
        name_list = name_list + names
        target_names = target_names+[gene_names[g]]* (attribution_all.shape[1])
    attribution_list = hstack( attribution_list)

    index_list = [f"{s}_{t}" for (s, t) in zip(name_list, target_names)]
    cou = pd.DataFrame({'index': index_list, 'source':name_list, 'target':target_names})
    cou = cou.set_index('index')
    return attribution_list, cou

def inference_one_target(
    target_gene,
    lrp_model,
    input_data,
    gene_names,
    background,
    xai_type='lrp-like',
    n_top_genes = 500,
    num_iterations = 10
   
):
    """
    Run inference for one target gene, masking its contributions multiple times.

    Args:
        target_gene: Gene to run inference for.
        lrp_model: The trained model for Layer-wise Relevance Propagation (LRP).
        data: Data, instance of CustomAnndataLoader.
        tf_gene_names: List of transcription factor gene names.
        lrp_process: Process to apply to LRP values ('abs' for absolute values).
        nt: Neural Taylor method ('lrp', 'smoothgrad', 'vargrad').
        masking_percentage: Percentage of input values to mask.
        num_iterations: Number of noisy perturbations to perform.

    Returns:
        A tuple containing:
            - Aggregated attribution values after multiple inferences with masking.
            - Names of transcription factors.
    """


    attributions_list = []
    for model in lrp_model:
        for _ in range(num_iterations):
            if xai_type == 'lrp-like':
                attribution = model.attribute(input_data, target=target_gene)
                    
            elif xai_type == 'shap-like':
                attribution = model.attribute(input_data, background, target = target_gene, n_samples = 1)
                #attribution_2 = model.attribute(input_data, background_2, target = target_gene, n_samples=1)
                diff = np.abs((attribution).detach().cpu().numpy())
                
            else:
                raise ValueError('No such method')

            attributions_list.append(diff)

    # Stack the attribution runs
    diff = np.stack(attributions_list)
    # Compute aggregate over stack
    diff = np.mean(attributions_list, axis=0)
    m2 = pd.DataFrame(diff).mean()
    idx = (m2).nlargest(n_top_genes).index
    diff = diff[:, idx]
    names = list(gene_names[idx])
    return diff, names




def draw_neg_binomial(mu, theta, size=1):
  """
  Draws random samples from a negative binomial distribution given mu and theta.

  Args:
    mu (float): The mean of the negative binomial distribution.
    theta (float): The dispersion parameter
    size (int or tuple of ints, optional): The number of samples to draw.
                                           Default is 1.

  Returns:
    ndarray or scalar: Drawn samples from the negative binomial distribution.
  """
  if mu <= 0 or theta <= 0:
    raise ValueError("mu and theta must be positive.")

  p = theta / (mu + theta)
  n = theta 

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

