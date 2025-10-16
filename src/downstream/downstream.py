import scanpy as sc
import anndata
from sklearn.cluster import SpectralClustering
import pandas as pd
import numpy as np
import scipy.sparse as scs

def downstream_recipe(adata)-> anndata.AnnData:
    """
    Downstream reciepe for an LRP anndata object:
    TODO: replace the config dict, to pass values
    """
    config = {'min_cells':1, 'n_neighbors': 30, 'leiden_resolution': 0.1, 'n_components': 100, 'knn_neighbors': 50}
    sc.pp.filter_genes(adata, min_cells=config['min_cells'])
    #sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
    #sc.pp.normalize_total(adata)
    #sc.pp.log1p(adata)

    sc.tl.pca(adata, svd_solver = 'randomized', zero_center = False)

    sc.pp.neighbors(adata, n_neighbors=config['knn_neighbors'])
    sc.tl.leiden(adata, resolution=config['leiden_resolution'])

    sc.tl.umap(adata, n_components = config['n_components'])
    return adata


def spectral_clustering(adata, n_clu = 2, key_added = 'spectral'):
    """
    Run sklearn spectral clustering on the neighbour matrix in the anndata object.

    Args:
    adata: Anndata object
    n_clu: Number of clusters to compute
    key_added: The key to add the new labelling to [Default: spectral]
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



def process(grn_adata, n_clu=2):
    if not scs.issparse(grn_adata.X):
        grn_adata.X[np.isnan(grn_adata.X) ] = 0
    grn_adata = downstream_recipe(grn_adata)
    print(f'clustering: {n_clu} clusters')
    grn_adata = spectral_clustering(grn_adata, n_clu=n_clu)
    return grn_adata
