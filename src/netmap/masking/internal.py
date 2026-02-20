import numpy as np
import pandas as pd
from scipy.sparse import issparse

import pandas as pd
import numpy as np
import itertools
import scanpy as sc

import scipy.stats as st
import itertools


def get_neighborhood_expression(adata, knn_neighbours =10, required_neighbours = 1, expression_threshold = 0):
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

    binary_expression = binarize_adata(adata, expression_threshold = expression_threshold)
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
    
    # 1. Preallocate the result matrix. 
    # Using np.int8 (1 byte) instead of default np.int64 (8 bytes) 
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
            
            # Use the 'out' parameter to write the multiplication result
            # directly into the preallocated column slice.
            
            np.multiply(
                binary_matrix[:, idx1], 
                binary_matrix[:, idx2], 
            )
            
    return result



def binarize_adata(adata, expression_threshold = 0):

    if issparse(adata.X):
        binary_expression = (adata.X.todense() > expression_threshold).astype(int)
    else:
        binary_expression = (adata.X > expression_threshold).astype(int)
    return binary_expression



def add_neighbourhood_expression_mask(adata, grn_adata, strict=False):
    """ Create a mask indicating whether the edge is likely actually
    expressed or not.

    Args:
        adata (_type_): _description_
        grn_adata (_type_): _description_

    Returns:
        _type_: _description_
    """

    if not strict:
        ne = get_neighborhood_expression(adata, required_neighbours=5)
    else:
        ne = binarize_adata(adata)
    mask = create_pairwise_binary_mask(ne, list(adata.var.index), grn_adata.var_names)
    
    grn_adata.layers['mask'] = mask
    grn_adata.var['count_nonzero'] = np.sum(grn_adata.layers['mask'], axis =0)
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