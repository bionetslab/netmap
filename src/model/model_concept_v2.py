import anndata as ad
from statsmodels.stats.nonparametric import rank_compare_2indep
import sys
sys.path.append('/data_nfs/og86asub/netmap/netmap-evaluation/netmap')

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

