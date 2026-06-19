import numpy as np
import pandas as pd
from scipy.sparse import issparse

import pandas as pd
import numpy as np
import itertools
import scanpy as sc

import scipy.stats as st
import itertools




def get_neighborhood_expression(adata, knn_neighbours =10, required_neighbours = 1, expression_threshold = 0, layer = 'X'):
    """
    Checks if each gene is expressed in the k-nearest neighbors (kNN) of each cell.

    Args:
        adata (anndata.AnnData): An AnnData object with a kNN graph in
                                 `adata.obsp['connectivities']`.

    Returns:
        pd.DataFrame: A cell x gene binary DataFrame where a value of 1 indicates
                      that the gene is expressed in at least one of the cell's
                      nearest neighbors, and 0 otherwise.
    """
    
    # Compute knn graph with low number of neighbours
    # in practice, the number may not be equal

    sc.pp.neighbors(adata, n_neighbors=knn_neighbours)
    # Get the connectivity matrix from the kNN graph
    connectivities = adata.obsp['connectivities'].copy()
    connectivities.data = np.ones(connectivities.data.shape)  # Binarize the graph

    binary_expression = binarize_adata(adata, expression_threshold = expression_threshold, layer = layer)
    # Perform matrix multiplication to check for neighbor expression
    # connectivities (cells x cells) @ binary_expression (cells x genes)
    # The result is a matrix where each value is the number of neighbors
    # expressing a given gene.
    neighbor_expression_counts = connectivities @ binary_expression

    # Binarize the result: 1 if the gene is expressed a required number of nieghbours
    neighborhood_expression = (neighbor_expression_counts >= required_neighbours).astype(int)


    return neighborhood_expression


def create_pairwise_binary_mask(binary_matrix, gene_list, ordered_pair_list):
    """
    Creates a dense 2D array of pairwise masks using preallocation.
    
    Args:
        binary_matrix (np.ndarray): Rows=cells, Cols=genes.
        gene_list (list): The names of the genes in binary_matrix columns.
        ordered_pair_list (list): The specific order of "GeneA_GeneB" strings 
                                  requested for the final output columns.
                                  
    Returns:
        np.ndarray: A 2D array of shape (num_cells, len(ordered_pair_list)).
    """
    num_cells = binary_matrix.shape[0]
    num_pairs = len(ordered_pair_list)

    binary_matrix = np.asarray(binary_matrix)

    result = np.zeros((num_cells, num_pairs), dtype=np.int8)
    

    # 2. Map gene names to their original column indices for O(1) lookup
    gene_to_idx = {name: i for i, name in enumerate(gene_list)}
    
    # 3. Fill the matrix column by column
    for col_idx, pair_str in enumerate(ordered_pair_list):
        g1_name, g2_name = pair_str.split('_')
        
        # Self-pairs (e.g., GeneA_GeneA) are skipped because 
        # the matrix is already initialized to zeros.
        if g1_name != g2_name:
            idx1 = gene_to_idx[g1_name]
            idx2 = gene_to_idx[g2_name]
            

            res = np.multiply(
                binary_matrix[:, idx1], 
                binary_matrix[:, idx2], 
            )
            result[:, col_idx] = np.array(res)
            
    return result



def binarize_adata(adata, expression_threshold = 0, layer = 'X'):

    if layer == 'X':
        if issparse(adata.X):
            binary_expression = (adata.X.todense() > expression_threshold).astype(int)
        else:
            binary_expression = (adata.X > expression_threshold).astype(int)
    else:
        if issparse(adata.layers[layer]):
            binary_expression = (adata.layers[layer].todense() > expression_threshold).astype(int)
        else:
            binary_expression = (adata.layers[layer] > expression_threshold).astype(int)
    return binary_expression



def add_neighbourhood_expression_mask(adata, grn_adata, strict=False, layer = 'X'):
    """ Create a mask indicating whether the edge is likely actually
    expressed or not.

    Args:
        adata (_type_): _description_
        grn_adata (_type_): _description_

    Returns:
        _type_: _description_
    """

    if 'X_pca' not in adata.obsm:
            raise KeyError(
                "adata.obsm['X_pca'] not found. Please run scanpy.pp.pca(adata) "
                "before calling this function to enable neighborhood calculations."
            )


    grn_genes = set()
    for pair in grn_adata.var_names:
        grn_genes.update(pair.split('_'))
    
    adata_genes = set(adata.var_names)
    missing_genes = grn_genes - adata_genes
    
    if missing_genes:

        print(f"Warning: {len(missing_genes)} genes from grn_adata are missing from adata.")
        if len(missing_genes) == len(grn_genes):
            raise ValueError("Zero overlap between GRN genes and adata.var_names.")
    
    if not strict:
        ne = get_neighborhood_expression(adata, required_neighbours=5, layer = layer)
    else:
        ne = binarize_adata(adata, layer = layer)
    mask = create_pairwise_binary_mask(ne, list(adata.var.index), grn_adata.var_names)
    
    grn_adata.layers['mask'] = mask
    grn_adata.var['count_nonzero'] = np.sum(grn_adata.layers['mask'], axis =0)
    return grn_adata


def add_cluster_wise_spearman(grn_adata, adata, cluster_column='leiden_remap'):
    """
    Compute cluster-wise Spearman rank correlation between source and target gene
    expression for each edge in grn_adata, adding a '{cluster}_spearman' column
    to grn_adata.var for every cluster.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData (obs=cells, var=edges).
            var must contain 'source' and 'target' columns.
        adata (anndata.AnnData): Expression AnnData (obs=cells, var=genes).
            obs must be aligned with grn_adata.obs.
        cluster_column (str): Column in grn_adata.obs with cluster labels.

    Returns:
        anndata.AnnData: grn_adata with new '{cluster}_spearman' columns in .var.
    """
    sources = grn_adata.var['source'].values
    targets = grn_adata.var['target'].values

    # Collect all unique genes, preserving order
    seen = {}
    for g in list(sources) + list(targets):
        seen[g] = None
    all_genes = list(seen.keys())

    adata_gene_set = set(adata.var_names)
    missing = [g for g in all_genes if g not in adata_gene_set]
    if missing:
        print(f"Warning: {len(missing)} GRN genes not found in adata and will produce NaN correlations.")
    available_genes = [g for g in all_genes if g in adata_gene_set]
    gene_to_idx = {g: i for i, g in enumerate(available_genes)}

    clusters = grn_adata.obs[cluster_column].unique()

    for ps in clusters:
        cell_mask = grn_adata.obs[cluster_column] == ps

        expr = adata[cell_mask, available_genes].X
        if issparse(expr):
            expr = expr.toarray()
        else:
            expr = np.asarray(expr)

        n_cells = expr.shape[0]

        if n_cells < 3:
            grn_adata.var[f'{ps}_spearman'] = np.nan
            continue

        # Rank-transform each gene column, then standardise
        ranked = np.apply_along_axis(st.rankdata, 0, expr)
        means = ranked.mean(axis=0)
        stds = ranked.std(axis=0, ddof=1)
        stds[stds == 0] = 1  # constant genes -> zero correlation
        rank_std = (ranked - means) / stds

        # Full Spearman correlation matrix via rank-based Pearson
        corr_matrix = (rank_std.T @ rank_std) / (n_cells - 1)

        spearman_vals = np.full(len(sources), np.nan)
        for i, (src, tgt) in enumerate(zip(sources, targets)):
            if src in gene_to_idx and tgt in gene_to_idx:
                spearman_vals[i] = corr_matrix[gene_to_idx[src], gene_to_idx[tgt]]

        grn_adata.var[f'{ps}_spearman'] = spearman_vals

    return grn_adata


def add_cluster_based_candidate_edges(grn_adata, cluster_column = 'leiden_remap', threshold = 0.5):
    vc = grn_adata.obs[cluster_column].value_counts()
    grn_adata.var[f'count_nonzero_norm'] = grn_adata.var[f'count_nonzero']/grn_adata.obs.shape[0]
    for ps in list(vc.index):
        grn_adata.var[f'{ps}_nonzero'] = (grn_adata[grn_adata.obs[cluster_column] == ps].layers['mask'].sum(axis = 0))/vc[ps]
        #grn_adata.var[f'{ps}_candidate_edge'] = np.abs(grn_adata.var[f'{ps}_nonzero']-grn_adata.var[f'count_nonzero_norm'])>0.1
        grn_adata.var[f'{ps}_candidate_edge'] = grn_adata.var[f'{ps}_nonzero']>threshold

    value_cols = [f'{ps}_candidate_edge' for ps in vc.index]
    grn_adata.var['candidate_edge'] = grn_adata.var[value_cols].sum(axis = 1)
    return grn_adata




def find_consistent_pairs(grn_adata, gene_names):
    """
    Creates a dictionary of binary masks for each cell and pair of genes,
    including both forward, reverse, and self-pairs (which are all zeros).

    Args:
        matrix_cells_x_genes (np.ndarray): A 2D numpy array where rows are cells
                                          and columns are genes.
        gene_list (list): A list of strings containing the names of the genes,
                          in the same order as the columns in the matrix.

    Returns:
        dict: A dictionary where keys are gene pair strings (e.g., 'GeneA_GeneB')
              and values are 1D numpy arrays representing the binary mask for that pair
              across all cells.
    """

    num_cells, num_edges = grn_adata.X.shape

    pairwise_mask_dict = {}

    gene_pairs_indices = list(itertools.combinations(gene_names, 2))
    for g1_idx, g2_idx in gene_pairs_indices:
        pairwise_mask_dict[f"{g1_idx}_{g2_idx}"] = st.spearmanr(grn_adata[:, [f"{g1_idx}_{g2_idx}", f"{g2_idx}_{g1_idx}"]].X).statistic
        # add reverse bc I am lazy
        pairwise_mask_dict[f"{g2_idx}_{g1_idx}"] = pairwise_mask_dict[f"{g1_idx}_{g2_idx}"]
    return pairwise_mask_dict