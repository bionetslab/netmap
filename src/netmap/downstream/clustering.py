import scanpy as sc
import anndata
from sklearn.cluster import SpectralClustering
import pandas as pd
import numpy as np
import scipy.sparse as scs
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import contingency_matrix



def downstream_recipe(grn_adata, config = {'min_cells':1, 'n_neighbors': 30, 'leiden_resolution': 0.1, 'n_components': 100, 'knn_neighbors': 50})-> anndata.AnnData:
    """
    Minimal downstream processing for GRN object. Runs sc.pca (without scaling),
    sc.neighbours, sc.leiden, sc. umap

    Parameters
    ----------
    grn_adata : AnnData
        A GRN anndata object to process

    config: dict
        Parameters for the processing

    Returns
    -------
    grn_adata : AnnData
        Processed Anndata object
    """
    
    sc.tl.pca(grn_adata, svd_solver = 'randomized', zero_center = False)
    sc.pp.neighbors(grn_adata, n_neighbors=config['knn_neighbors'])
    sc.tl.leiden(grn_adata, resolution=config['leiden_resolution'])
    sc.tl.umap(grn_adata, n_components = config['n_components'])

    return grn_adata


def spectral_clustering(adata, n_clu = 2, key_added = 'spectral'):
    """
    Run sklearn spectral clustering on the neighbour matrix in the anndata object.


    Parameters
    ----------
    grn_adata : AnnData
        A GRN anndata object to process

    n_clu: int
        Number of clusters to compute
    key_added: str
        The key to add the new labelling to [Default: spectral]

    Returns
    -------
    grn_adata : AnnData
        Clustered Anndata object with an additional column in the .obs
        slot of the object named `key_added`
    """

    sc.pp.neighbors(adata)
    ssc = SpectralClustering(n_clusters=n_clu,assign_labels='discretize',random_state=0, affinity= 'precomputed_nearest_neighbors').fit(adata.obsp['distances'])
    
    counter = 0
    key_added_t = key_added
    while key_added_t in adata.obs.columns:
        counter = counter + 1
        key_added_t = f'{key_added}_{counter}'
    adata.obs[key_added_t] = ssc.labels_
    adata.obs[key_added_t] = pd.Categorical(adata.obs[key_added_t])
    return adata



def process(grn_adata, n_clu=2, key_added = 'spectral'):
    """
    Wrapper function for convenience processing and clustering

    Parameters
    ----------
    grn_adata : AnnData
        A GRN anndata object to process

    n_clu: int
        Number of clusters to compute
    key_added: str
        The key to add the new labelling to [Default: spectral]

    Returns
    -------
    grn_adata : AnnData
        Processed Anndata object with an additional column in the .obs
        slot of the object named `key_added`
    """

    if not scs.issparse(grn_adata.X):
        grn_adata.X[np.isnan(grn_adata.X) ] = 0
    grn_adata = downstream_recipe(grn_adata)
    grn_adata = spectral_clustering(grn_adata, n_clu=n_clu, key_added = key_added)
    return grn_adata



def unify_group_labelling(adata, grn_adata, col_adata, col_grn_adata) -> float:


    """
    Adjust group labelling such that grn_adata has the same group label than the 
    corresponding column in adata based on the grn column.

    Tee prrocessed GRN  Anndata object wwill contian ith an additional column in the
    .obs slot containing a '_remap' column which contains a relabelled column
    matching the clustering in adata

    Returns a score of the matching as the cost of the matching divided by the number
    of cells. The score ranges between 1/n_clusters and 1, with 1 indicating an identical
    clustering and 1/n_clu a random matching.
    
    Parameters
    ----------
    adata : AnnData
        original Anndata object containing a clustering column in the
        obs object specified as `col_adata`
    grn_adata : AnnData
        original Anndata object containing a clustering column in the
        obs object specified as `col_grn_adata`

    col_adata_clu: str
        Name of the reference column in expression object
    col_grn_adata: str
        Name of the novel clustering column in GRN object

    Returns
    -------
    score: int 
        The score of the linear sum assignment divided by the number of cells.
    """

    cm = contingency_matrix(adata.obs[col_adata], grn_adata.obs[col_grn_adata])
    row_ind, col_ind = linear_sum_assignment(cm, maximize = True)
    
    names_ad = np.unique(adata.obs[col_adata])
    names_grn = np.unique(grn_adata.obs[col_grn_adata])
    mapping = {}
    reverse_mapping = {}
    for i in range(len(row_ind)):
        reverse_mapping[names_grn[col_ind[i]]] = names_ad[row_ind[i]]

    col_grn_adata_remapped = col_grn_adata + '_remap'
    if isinstance(np.unique(grn_adata.obs[col_grn_adata])[0], str):
        grn_adata.obs[col_grn_adata_remapped] = [reverse_mapping[a] for a in grn_adata.obs[col_grn_adata]]
    else:
        grn_adata.obs[col_grn_adata_remapped] = [reverse_mapping[int(a)] for a in grn_adata.obs[col_grn_adata]]

    grn_adata.obs[col_grn_adata_remapped] = pd.Categorical(grn_adata.obs[col_grn_adata_remapped])

    score = cm[row_ind, col_ind].sum()/adata.obs.shape[0]
    
    return score


