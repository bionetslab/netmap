from sklearn.cluster import SpectralClustering

import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import hstack

import numpy as np
from scipy.stats import nbinom
from sklearn.cluster import SpectralClustering
import cuml
import pandas as pd
import numpy as np
from tqdm import tqdm
import rapids_singlecell as rsc
from numpy import hstack
import anndata as ad
from captum.attr import GradientShap

from utils.data_utils import attribution_to_anndata

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
        #for _ in range(num_iterations):
        if xai_type == 'lrp-like':
            attribution = model.attribute(input_data, target=target_gene)
                
        elif xai_type == 'shap-like':
            attribution = model.attribute(input_data, background, target = target_gene, n_samples = 10)
            
            my_att = ad.AnnData(attribution.detach().cpu().numpy())
            rsc.get.anndata_to_GPU(my_att) # moves `.X` to the GPU
            rsc.pp.scale(my_att)
            rsc.pp.pca(my_att)
            rsc.pp.neighbors(my_att)
            
            sclu = SpectralClustering( n_clusters=2)
            sclu = sclu.fit(my_att.obsp['connectivities'])
            my_att.obs[f'clu_{target_gene}'] = pd.Categorical(sclu.labels_)

            
        else:
            raise ValueError('No such method')


    m2 = pd.DataFrame(my_att.X.get()).mean()
    idx = (m2).nlargest(n_top_genes).index
    my_att = my_att[:, idx]
    names = list(gene_names[idx])
    return my_att, names


from numpy import vstack
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
    clustering_list = []
    for g in tqdm(range(data.shape[1])):
        attribution_all, names = inference_one_target(g, lrp, data, gene_names, background, 
                                                        xai_type=xai_type, n_top_genes = n_top_genes, num_iterations=num_iterations)
        
        attribution_list.append(attribution_all.X)
        clustering_list.append(attribution_all.obs[f'clu_{g}'])
        name_list = name_list + names
        target_names = target_names+[gene_names[g]]* (attribution_all.shape[1])
    attribution_list = hstack( attribution_list)
    clustering_list = hstack(clustering_list)
    index_list = [f"{s}_{t}" for (s, t) in zip(name_list, target_names)]
    cou = pd.DataFrame({'index': index_list, 'source':name_list, 'target':target_names})
    cou = cou.set_index('index')
    return attribution_list, cou, clustering_list


def compute_attributions(model_zoo, data_tensor, latent, gene_names, backgrounds, adata, n_top = 250):
    myexplainers = [GradientShap(mo, multiply_by_inputs=False) for mo in model_zoo]
    attributions = []
    for b in backgrounds.keys():
        aggregated_attribution, cou, clusterings = inference_complete(myexplainers, data_tensor[latent.obs['consensus'] == b].cuda(), gene_names, backgrounds[b], xai_type='shap-like', num_iterations=5,n_top_genes=n_top )
        ad = attribution_to_anndata(aggregated_attribution, var=cou, obs=latent.obs[latent.obs['consensus'] == b])
        attributions.append(ad)
    return attributions