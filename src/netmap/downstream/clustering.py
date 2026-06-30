"""Dimensionality reduction, clustering, and label-matching utilities for GRN AnnData.

The main pipeline entry :func:`process` runs PCA → kNN → Leiden → UMAP then
adds spectral clustering labels.  :func:`unify_group_labelling` remaps GRN
cluster labels to match a reference expression object's clusters.
"""

import scanpy as sc
import anndata
from sklearn.cluster import SpectralClustering
import pandas as pd
import numpy as np
import scipy.sparse as scs
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import contingency_matrix

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching



def downstream_recipe(grn_adata, **kwargs) -> sc.AnnData:
    """Run the standard Scanpy dimensionality-reduction and clustering pipeline.

    Applies PCA, kNN graph construction, Leiden clustering, and UMAP embedding
    to ``grn_adata`` in sequence.  All parameters have defaults and can be
    overridden via keyword arguments.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData to process in-place.
        **kwargs: Optional parameter overrides:

            - ``n_neighbors`` (int): Leiden kNN neighbours. Default 30.
            - ``leiden_resolution`` (float): Leiden resolution. Default 0.1.
            - ``n_components`` (int): UMAP output dimensions. Default 100.
            - ``knn_neighbors`` (int): Neighbours for ``sc.pp.neighbors``. Default 50.
            - ``svd_solver`` (str): PCA solver. Default ``'randomized'``.

    Returns:
        anndata.AnnData: ``grn_adata`` with PCA, kNN, Leiden, and UMAP results
            added.
    """
    # Define defaults and update with any provided kwargs
    config = {
        'n_neighbors': 30,
        'leiden_resolution': 0.1,
        'n_components': 100,
        'knn_neighbors': 50,
        'svd_solver': 'randomized'
    }
    config.update(kwargs)

    # Use the config values in the scanpy functions
    sc.tl.pca(grn_adata, svd_solver=config['svd_solver'], zero_center=False)

    # Note: sc.pp.neighbors uses n_neighbors.
    # If you specifically want to use your 'knn_neighbors' key:
    sc.pp.neighbors(grn_adata, n_neighbors=config['knn_neighbors'])

    sc.tl.leiden(grn_adata, resolution=config['leiden_resolution'])

    # n_components in UMAP usually refers to the dimensions of the embedding (default 2)
    sc.tl.umap(grn_adata, n_components=config['n_components'])

    return grn_adata

def process(grn_adata, n_clu=2, key_added='spectral', **kwargs):
    """Prepare GRN AnnData and run the full clustering pipeline.

    Fills NaN values in ``grn_adata.X`` with 0, calls :func:`downstream_recipe`,
    then adds spectral clustering via :func:`spectral_clustering`.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData to cluster.
        n_clu (int): Number of spectral clusters. Defaults to 2.
        key_added (str): Column name for spectral labels in ``.obs``.
            Defaults to ``'spectral'``.
        **kwargs: Forwarded to :func:`downstream_recipe`.

    Returns:
        anndata.AnnData: Clustered GRN AnnData.
    """

    if not scs.issparse(grn_adata.X):
        grn_adata.X[np.isnan(grn_adata.X)] = 0

    # Pass the kwargs directly into the recipe
    grn_adata = downstream_recipe(grn_adata, **kwargs)

    # Assuming spectral_clustering is defined elsewhere
    grn_adata = spectral_clustering(grn_adata, n_clu=n_clu, key_added=key_added)

    return grn_adata


def spectral_clustering(adata, n_clu=2, key_added='spectral'):
    """Run scikit-learn spectral clustering on the neighbour graph of an AnnData object.

    Recomputes the neighbour graph with default parameters via
    ``sc.pp.neighbors``, then fits a
    :class:`sklearn.cluster.SpectralClustering` model using the precomputed
    nearest-neighbour distances stored in ``adata.obsp['distances']``.  The
    resulting integer cluster labels are stored as a categorical column in
    ``adata.obs``.  If ``key_added`` already exists in ``adata.obs`` a
    numeric suffix (``_1``, ``_2``, …) is appended to avoid overwriting
    existing annotations.

    Args:
        adata (anndata.AnnData): AnnData object on which to run spectral
            clustering.  Modified in-place.
        n_clu (int): Number of clusters to compute.  Defaults to 2.
        key_added (str): Base name of the column added to ``adata.obs``.
            Defaults to ``'spectral'``.

    Returns:
        anndata.AnnData: The input ``adata`` object with an additional
        categorical column in ``.obs`` named ``key_added`` (or
        ``key_added_<n>`` if the name was already taken) containing the
        spectral cluster labels.
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


def unify_group_labelling(adata, grn_adata, col_adata, col_grn_adata) -> float:
    """Adjust GRN cluster labels to match a reference clustering via optimal assignment.

    Uses the Hungarian algorithm (:func:`scipy.optimize.linear_sum_assignment`)
    to find a one-to-one mapping from GRN cluster labels to reference cluster
    labels that maximises the number of correctly matched cells.  The remapped
    labels are stored as a new categorical column named
    ``<col_grn_adata>_remap`` in ``grn_adata.obs``.

    Args:
        adata (anndata.AnnData): Reference AnnData object containing a
            clustering column in ``obs`` specified by ``col_adata``.
        grn_adata (anndata.AnnData): GRN AnnData object containing a
            clustering column in ``obs`` specified by ``col_grn_adata``.
            Modified in-place by adding the ``<col_grn_adata>_remap`` column.
        col_adata (str): Name of the reference clustering column in
            ``adata.obs``.
        col_grn_adata (str): Name of the GRN clustering column in
            ``grn_adata.obs`` to be remapped.

    Returns:
        float: Score of the linear-sum assignment divided by the number of
            cells.  Ranges between ``1/n_clusters`` (random matching) and
            ``1.0`` (identical clustering).
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




def unify_group_labelling(adata, grn_adata, col_adata, col_grn_adata, return_mapping=True):
    """Match overclustered GRN labels to a reference clustering using argmax assignment.

    Each GRN cluster is mapped to the reference cluster that contains the majority
    of its cells (argmax of the contingency matrix column).  Unlike the Hungarian
    algorithm, this allows many-to-one mappings, making it suitable when
    ``grn_adata`` has more clusters than ``adata``.

    Args:
        adata (anndata.AnnData): Reference AnnData with ground-truth cluster labels.
        grn_adata (anndata.AnnData): GRN AnnData whose clusters will be remapped.
        col_adata (str): Column in ``adata.obs`` with reference cluster labels.
        col_grn_adata (str): Column in ``grn_adata.obs`` with GRN cluster labels.
        return_mapping (bool): If ``True`` also return the mapping dict.
            Defaults to ``True``.

    Returns:
        float or tuple: Fraction of correctly matched cells.  If
            ``return_mapping=True``, returns ``(score, reverse_mapping)``.
    """
    # 1. Compute Contingency Matrix
    # Rows = adata (Ref), Cols = grn_adata (Target)
    cm = contingency_matrix(adata.obs[col_adata], grn_adata.obs[col_grn_adata])

    names_ad = np.unique(adata.obs[col_adata])
    names_grn = np.unique(grn_adata.obs[col_grn_adata])

    cost_matrix = cm.max() - cm.T

    reverse_mapping = {}
    total_matched_cells = 0

    # Efficiently find the max for each GRN cluster
    best_ref_indices = np.argmax(cm, axis=0)

    for grn_idx, ref_idx in enumerate(best_ref_indices):
        grn_label = names_grn[grn_idx]
        ref_label = names_ad[ref_idx]
        reverse_mapping[grn_label] = ref_label
        total_matched_cells += cm[ref_idx, grn_idx]

    # 3. Apply Mapping
    col_grn_remapped = f"{col_grn_adata}_remap"
    grn_adata.obs[col_grn_remapped] = grn_adata.obs[col_grn_adata].map(reverse_mapping)
    grn_adata.obs[col_grn_remapped] = pd.Categorical(grn_adata.obs[col_grn_remapped])

    score = total_matched_cells / grn_adata.n_obs

    if return_mapping:
        return score, reverse_mapping
    return score
