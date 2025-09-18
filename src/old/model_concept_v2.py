import anndata as ad
from statsmodels.stats.nonparametric import rank_compare_2indep
import sys
sys.path.append('/data_nfs/og86asub/netmap/netmap-evaluation/netmap')

import scanpy as sc
import time 

from netmap.src.utils.misc import write_config

from netmap.src.model.nbautoencoder import *
import scanpy as sc

from sklearn.model_selection import train_test_split
import time
from captum.attr import GradientShap, LRP
from netmap.src.old.inferrence_simple import *
from netmap.src.utils.data_utils import attribution_to_anndata
from netmap.src.old.pipeline import *
import numpy as np


from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN

import os.path as op
import os

def get_explainer(trained_model, lrp='GradientShap'):
    if lrp == 'GradientShap':
        explainer = GradientShap(trained_model)
        xai_type = 'shap-like'
    elif lrp == 'LRP':
        explainer = LRP(trained_model)
        xai_type = 'lrp-like'
    
    return explainer, xai_type



def create_model_zoo(data_tensor, n_models = 4, n_epochs = 500):
    model_zoo = []
    for _ in range(n_models):

        data_train2, data_test2 = train_test_split(data_tensor,test_size=0.01, shuffle=True)

        trained_model2 = NegativeBinomialAutoencoder(input_dim=data_tensor.shape[1], latent_dim=10, dropout_rate = 0.02)
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


def inference_one_target(
    target_gene,
    lrp_model,
    input_data,
    background,
    xai_type='lrp-like',
    top_perc = False
   
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
    for m in range(len(lrp_model)):
        model = lrp_model[m]
        #for _ in range(num_iterations):
        if xai_type == 'lrp-like':
            attribution = model.attribute(input_data)
                
        elif xai_type == 'shap-like':
            attribution = model.attribute(input_data, baselines = background, target = target_gene)

        attributions_list.append(attribution.detach().cpu().numpy())

    a = np.mean(attributions_list, axis = 0)
    my_att = ad.AnnData(a)
    #rsc.get.anndata_to_GPU(my_att) # moves `.X` to the GPU
    sc.pp.scale(my_att)
    try:
        sc.pp.pca(my_att,n_comps=25 )
    except:
        try:
            sc.pp.pca(my_att,n_comps=25 )
        except:
            return False, None, None


    sc.pp.neighbors(my_att, n_neighbors=15)
    sc.tl.leiden(my_att, resolution=0.05)
    #rsc.get.anndata_to_CPU(my_att)


    #sc.tl.umap(my_att)
    #sc.pl.umap(my_att, color='leiden')

    sig_dict = {}

    genelist = []
    if len(np.unique(my_att.obs['leiden']))>1 and not top_perc:
        for cat in np.unique(my_att.obs['leiden']):
            statisi =rank_compare_2indep(x1=a[my_att.obs['leiden']==cat], x2= a[my_att.obs['leiden']!=cat])
            sig_and_high = np.where((statisi.pvalue<(0.01/(input_data.shape[1]*input_data.shape[1])))  & (statisi.prob1>= 0.9))
            sig_dict[cat] = sig_and_high
            genelist = genelist+ list(sig_and_high[0])

    
    else:
        print('ALL GENESq')
        m = np.abs(a).mean(axis = 0)
        genelist = np.where(m>np.percentile(m, 0))[0]


    

    genelist = np.array(genelist).flatten()
    return my_att[:, genelist], sig_dict, genelist


def wrapper(models, data_train_full_tensor, gene_names, top_perc =False):
    tms = []
    name_list = []
    grn_anndatas = []
    target_names = []
    clusterings = {}
    for trained_model in models:        
        trained_model.forward_mu_only = True
        explainer, xai_type = get_explainer(trained_model, 'GradientShap')
        tms.append(explainer)

    #for g in range(data_train_full_tensor.shape[1]):
    for g in range(data_train_full_tensor.shape[1]):

        my_att, sig_dict, genelist =  inference_one_target( 
            g,
            tms,
            data_train_full_tensor,
            data_train_full_tensor,
            xai_type=xai_type,
            top_perc=top_perc)
    

        name_list = name_list + list(gene_names[genelist])
        target_names = target_names+[gene_names[g]]* (my_att.X.shape[1])
        

        clusterings[f'T_{gene_names[g]}'] = np.array(my_att.obs['leiden'])
        grn_anndatas.append(my_att.X)


    grn_anndatas = hstack(grn_anndatas)
    index_list = [f"{s}_{t}" for (s, t) in zip(name_list, target_names)]
    cou = pd.DataFrame({'index': index_list, 'source':name_list, 'target':target_names})
    cou = cou.set_index('index')

    clusterings = pd.DataFrame(clusterings)

    grn_adata = attribution_to_anndata(grn_anndatas, var=cou, obs = clusterings)

    return grn_adata


def run_netmap(config, dataset_config):

    print('Version 2')
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
        data_tensor = torch.tensor(data_tensor, dtype=torch.float32)


    print(data_tensor.shape)

    model_zoo = create_model_zoo(data_tensor, n_models=config.n_models, n_epochs=500)
    grn_adata = wrapper(model_zoo, data_tensor.cuda(), gene_names, config.top_perc)

    adob = adata.obs.reset_index()
    grn_adata.obs['cell_id'] = np.array(adob['cell_id'])
    grn_adata.obs['grn'] = np.array(adob['grn'])

    
    model_elapsed = time.monotonic()-model_start
    grn_adata.write_h5ad(op.join(config.output_directory,config.adata_filename))

    time_elapsed_total = time.monotonic()-start_total


    res = {'time_elapsed_total': time_elapsed_total, 'time_elapsed_netmap': model_elapsed} 
    write_config(res, file=op.join(config.output_directory, 'results.yaml'))




import numpy as np
from captum.attr import GuidedBackprop, GradientShap, Deconvolution
import pingouin as pingu
import torch
from
import pandas as pd

from netmap.src.utils.data_utils import attribution_to_anndata



def quantile_partitioning(data: np.ndarray, q: int) -> np.ndarray:
    """
    Performs quantile partitioning on a 1D NumPy array.

    The method orders the data and divides it into 'q' equal-sized partitions.
    A new array (mask) is created where each element is assigned a value
    of k/q, where k is the quantile that the element belongs to.

    Args:
        data (np.ndarray): A 1D NumPy array of numerical data.
        q (int): The number of quantiles to partition the data into. Must be a
                 positive integer.

    Returns:
        np.ndarray: A new array of the same shape as the input data, with
                    values representing the quantile partition.

    Raises:
        ValueError: If q is not a positive integer or if the input data is not
                    a 1D NumPy array.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input data must be a 1D NumPy array.")
    if not isinstance(q, int) or q <= 0:
        raise ValueError("The number of quantiles 'q' must be a positive integer.")

    n = len(data)
    if n == 0:
        return np.array([])
    
    # 1. Order data using argsort to get the indices that would sort the array
    sorted_indices = np.argsort(data)
    
    # 2. Cut into q equal pieces by calculating the size of each partition
    # We use float division to handle cases where n is not perfectly divisible by q.
    partition_size = n / q
    
    # 3. Initialize mask with same dimension as data
    mask = np.zeros_like(data, dtype=float)
    
    # 4. With k=current quantile add k/q to all cells belonging to k
    for k in range(1, q + 1):
        start_index = int((k - 1) * partition_size)
        end_index = int(k * partition_size)
        
        # Get the original indices that belong to the current quantile
        quantile_indices = sorted_indices[start_index:end_index]
        
        # Assign the value k/q to the corresponding positions in the mask
        mask[quantile_indices] = k / q
        
    return mask

def quantile_partitioning_2d(data: np.ndarray, q: int) -> np.ndarray:
    """
    Performs quantile partitioning row-wise on a 2D NumPy array using
    np.apply_along_axis for efficiency.

    Args:
        data (np.ndarray): A 2D NumPy array of numerical data.
        q (int): The number of quantiles to partition each row into.

    Returns:
        np.ndarray: A new 2D array with the same shape as the input,
                    where each row contains the quantile partitions.

    Raises:
        ValueError: If the input data is not a 2D NumPy array.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")

    # Use np.apply_along_axis to apply the 1D function to each row (axis=1).
    return np.apply_along_axis(quantile_partitioning, 1, data, q)

def compute_correlation_metric(data, cor_type):
    # Compute gene correlation measure
    #  'pingouin.pcorr', 'np.cov', 'np.corcoeff'
    if cor_type ==  'pingouin.pcorr':
        cov = pingu.pcorr(pd.DataFrame(data))
    elif cor_type == 'np.cov':
        cov = np.cov(data.T)
    elif cor_type == 'np.corrcoeff':
        cov = np.corrcoef(data.T)
    elif cor_type == 'None':
        cov = 1
    else: 
        cov = 1
    return cov

def aggregate_attributions(attributions, strategy = 'mean'):
    if strategy == 'mean':
        return np.mean(attributions, axis = 0)
    elif strategy == 'sum':
        return np.sum(attributions, axis = 0)
    elif strategy == 'median':
        return np.median(attributions, axis = 0)
    else:
        # Default to mean aggregation
        return np.mean(attributions, axis = 0)
    

def get_explainer(model, explainer_type, raw=False):
    if explainer_type in ['GuidedBackprop', 'Deconvolution']:
        explainer_mode = 'lrp-like'
    else:
        explainer_mode = 'shap-like'
    
        
    if explainer_type == 'GuidedBackprop': #fast
        explainer = GuidedBackprop(model)
    elif explainer_type == 'GradientShap': #fast
        if raw:
            explainer = GradientShap(model, multiply_by_inputs=False)
        else:
            explainer = GradientShap(model, multiply_by_inputs=True)

    elif explainer_type == 'Deconvolution': #fast
        explainer = Deconvolution(model)
    else:
        raise ValueError('no such method')
        
    return explainer, explainer_mode



def shuffle_each_column_independently(tensor):
    """
    Shuffles each column of a 2D PyTorch tensor independently.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: A new tensor with each of its columns independently shuffled.
    """
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional to shuffle columns.")

    # Create an empty tensor of the same size to store the shuffled columns
    shuffled_tensor = torch.empty_like(tensor)

    # Iterate through each column, shuffle it, and place it in the new tensor
    for i in range(tensor.size(1)):
        column = tensor[:, i]
        idx = torch.randperm(column.nelement())
        shuffled_tensor[:, i] = column[idx]

    return shuffled_tensor

def attribution_one_target( 
        target_gene,
        lrp_model,
        input_data,
        background,
        xai_type='lrp-like',
        randomize_background = False):
    
    attributions_list = []
    for m in range(len(lrp_model)):
        # Randomize backgorund for each round
        if randomize_background:
            background = shuffle_each_column_independently(background)

        model = lrp_model[m]
        #for _ in range(num_iterations):
        if xai_type == 'lrp-like':
            #print(input_data)
            #print(target_gene)
            attribution = model.attribute(input_data, target=target_gene)
                
        elif xai_type == 'shap-like':
            attribution = model.attribute(input_data, baselines = background, target = target_gene)

        attributions_list.append(attribution.detach().cpu().numpy())
    return attributions_list


def inferrence(models, data_train_full_tensor, gene_names, config):

    data = data_train_full_tensor.detach().cpu().numpy()
    tms = []
    name_list = []
    target_names = []
    
    
    for trained_model in models:        
        trained_model.forward_mu_only = True
        explainer, xai_type = get_explainer(trained_model, config.xai_method, config.raw_attribution)
        tms.append(explainer)

    attributions = []
    ## ATTRIBUTIONS
    for g in tqdm(range(data_train_full_tensor.shape[1])):
    #for g in range(2):

        attributions_list = attribution_one_target(
            g,
            tms,
            data_train_full_tensor,
            data_train_full_tensor,
            xai_type=xai_type,
            randomize_background = True)
        attributions.append(attributions_list)

        ## AGGREGATION: REPLACE LIST BY AGGREGATED DATA
    quantiles = []
    for i in range(len(attributions)):
        current_quantiles = []
        for j in range(len(attributions[i])):
        # CURRENTLY MEAN
            current_quantiles.append(quantile_partitioning_2d(attributions[i][j], q = 10))
        quantiles.append(current_quantiles)
    

    ## AGGREGATION: REPLACE LIST BY AGGREGATED DATA
    for i in range(len(attributions)):
        # CURRENTLY MEAN
        attributions[i] = aggregate_attributions(attributions[i], strategy=config.aggregation_strategy )
        quantiles[i] = aggregate_attributions(quantiles[i], strategy='sum')
    
    print(attributions)
    ## PENALIZE:
    if config.penalty != 'None':
        penalty_matrix = compute_correlation_metric(data, cor_type=config.penalty)
        for i in range(len(attributions)):
            # CURRENTLY MEAN
            attributions[i] = np.dot(attributions[i], (1-penalty_matrix))

    print(attributions)

    attributions = np.hstack(attributions)
    counter = np.hstack(quantiles)
    
    index_list = [f"{s}_{t}" for (s, t) in zip(name_list, target_names)]
    cou = pd.DataFrame({'index': index_list, 'source':name_list, 'target':target_names})
    cou = cou.set_index('index')

    grn_adata = attribution_to_anndata(attributions, var=cou)
    grn_adata.layers['quantile_count'] = counter

    return grn_adata
