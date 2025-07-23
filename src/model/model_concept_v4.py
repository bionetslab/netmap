import sys
sys.path.append('/data_nfs/og86asub/netmap/netmap-evaluation/')

import scanpy as sc
import time 

from netmap.src.utils.misc import write_config

from netmap.src.model.negbinautoencoder import *
import scanpy as sc

from sklearn.model_selection import train_test_split
import time
from captum.attr import GradientShap, LRP
from netmap.src.model.inferrence_simple import *
from netmap.src.utils.data_utils import attribution_to_anndata
from netmap.src.model.pipeline import *
import numpy as np


from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN

import os.path as op
import os

import anndata as ad
from statsmodels.stats.nonparametric import rank_compare_2indep

import numpy as np
import pandas as pd
import scipy.sparse as scs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from captum.attr import *
import pingouin as pingu

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

def attribution_one_target( 
        target_gene,
        lrp_model,
        input_data,
        background,
        xai_type='lrp-like'):
    
    attributions_list = []
    for m in range(len(lrp_model)):
        model = lrp_model[m]
        #for _ in range(num_iterations):
        if xai_type == 'lrp-like':
            attribution = model.attribute(input_data, target=target_gene)
                
        elif xai_type == 'shap-like':
            attribution = model.attribute(input_data, baselines = background, target = target_gene)

        attributions_list.append(attribution.detach().cpu().numpy())
    return attributions_list

def get_differential_edges(attribution_anndata, percentile = 10):
    genelist = []
    if len(np.unique(attribution_anndata.obs['leiden']))>1 :
        for cat in np.unique(attribution_anndata.obs['leiden']):
            statisi =rank_compare_2indep(x1=attribution_anndata.X[attribution_anndata.obs['leiden']==cat], x2= attribution_anndata.X[attribution_anndata.obs['leiden']!=cat])
            sig_and_high = np.where((statisi.pvalue<(0.01/(attribution_anndata.X.shape[1]*attribution_anndata.X.shape[1])))  & (statisi.prob1>= 0.9))
            genelist = genelist+ list(sig_and_high[0])
            # Combine the two arrays of indices and sort them
        genelist = np.unique(np.sort(genelist))

    else:
        # FALLBACk
        m = np.abs(attribution_anndata.X).mean(axis=0)
        # Get the indices of genes in the top 10%
        top_10_percent_indices = np.where(m > np.percentile(m, 100-percentile))[0]

        # Get the indices of genes in the bottom 10%
        bottom_10_percent_indices = np.where(m < np.percentile(m, percentile))[0]

        # Combine the two arrays of indices and sort them
        genelist = np.unique(np.sort(
            np.concatenate((top_10_percent_indices, bottom_10_percent_indices))
        ))
    return genelist

def get_percentile_edges(attribution_anndata, percentile = 10):
    # FALLBACk
    m = attribution_anndata.X.mean(axis=0)
    # Get the indices of genes in the top 10%
    top_10_percent_indices = np.where(m > np.percentile(m, 100-percentile))[0]

    # Get the indices of genes in the bottom 10%
    bottom_10_percent_indices = np.where(m < np.percentile(m, percentile))[0]

    # Combine the two arrays of indices and sort them
    genelist = np.sort(
        np.concatenate((top_10_percent_indices, bottom_10_percent_indices))
    )
    return genelist

def get_edges(attribution_anndata, use_differential=False, percentile = 10):
    if use_differential:
        return get_differential_edges(attribution_anndata, percentile=percentile)
    else:
        return get_percentile_edges(attribution_anndata, percentile=percentile)
    
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
    
def wrapper(models, data_train_full_tensor, gene_names, config):

    data = data_train_full_tensor.detach().cpu().numpy()
    tms = []
    name_list = []
    target_names = []
    clusterings = {}
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
            xai_type=xai_type)

        attributions.append(attributions_list)



    ## AGGREGATION: REPLACE LIST BY AGGREGATED DATA
    for i in range(len(attributions)):
        # CURRENTLY MEAN
        attributions[i] = aggregate_attributions(attributions[i], strategy=config.aggregation_strategy )
    
    ## PENALIZE:
    if config.penalty != 'None':
        penalty_matrix = compute_correlation_metric(data, cor_type=config.penalty)
        for i in range(len(attributions)):
            # CURRENTLY MEAN
            attributions[i] = np.dot(attributions[i], (1-penalty_matrix))


    ## CLUSTERING: CLUSTER EACH TARGET INDVIDUALLY
    for i in range(len(attributions)):
 
        attributions[i] = ad.AnnData(attributions[i])
        sc.pp.scale(attributions[i])
        try:
            sc.pp.pca(attributions[i],n_comps=50)
        except:
            try:
                sc.pp.pca(attributions[i],n_comps=50 )
            except:
                return False, None, None
            
        sc.pp.neighbors(attributions[i], n_neighbors=15)
        sc.tl.leiden(attributions[i], resolution=0.1)

        clusterings[f'T_{gene_names[i]}'] = np.array(attributions[i].obs['leiden'])

    
    #EDGE SELECTION:
    for i in range(len(attributions)):
        edge_indices = get_edges(attributions[i], use_differential=config.use_differential, percentile=config.percentile)
        name_list = name_list + list(gene_names[edge_indices])
        target_names = target_names+[gene_names[i]]* len(edge_indices)
        attributions[i] = attributions[i][:,edge_indices].X

    attributions = np.hstack(attributions)
    
    index_list = [f"{s}_{t}" for (s, t) in zip(name_list, target_names)]
    cou = pd.DataFrame({'index': index_list, 'source':name_list, 'target':target_names})
    cou = cou.set_index('index')

    clusterings = pd.DataFrame(clusterings)

    grn_adata = attribution_to_anndata(attributions, var=cou, obs = clusterings)

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
    grn_adata = wrapper(model_zoo, data_tensor.cuda(), gene_names, config)

    adob = adata.obs.reset_index()
    grn_adata.obs['cell_id'] = np.array(adob['cell_id'])
    grn_adata.obs['grn'] = np.array(adob['grn'])

    
    model_elapsed = time.monotonic()-model_start
    grn_adata.write_h5ad(op.join(config.output_directory,config.adata_filename))

    time_elapsed_total = time.monotonic()-start_total


    res = {'time_elapsed_total': time_elapsed_total, 'time_elapsed_netmap': model_elapsed} 
    write_config(res, file=op.join(config.output_directory, 'results.yaml'))

