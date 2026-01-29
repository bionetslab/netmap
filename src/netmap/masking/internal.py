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


def create_pairwise_binary_mask(binary_matrix, gene_list):
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

    num_cells, num_genes = binary_matrix.shape

    if len(gene_list) != num_genes:
        raise ValueError("The length of the gene_list must match the number of genes (columns) in the matrix.")

    pairwise_mask_dict = {}
    zero_vector = np.zeros(num_cells, dtype=int)
    for g_idx, gene_name in enumerate(gene_list):
        key = f"{gene_name}_{gene_name}"
        pairwise_mask_dict[key] = zero_vector

    gene_pairs_indices = list(itertools.combinations(range(num_genes), 2))
    for g1_idx, g2_idx in gene_pairs_indices:
        mask = binary_matrix[:, g1_idx] * binary_matrix[:, g2_idx]
        key_fwd = f"{gene_list[g1_idx]}_{gene_list[g2_idx]}"
        pairwise_mask_dict[key_fwd] = mask
        key_rev = f"{gene_list[g2_idx]}_{gene_list[g1_idx]}"
        pairwise_mask_dict[key_rev] = mask

    return pairwise_mask_dict



def dict_to_dataframe(mask_dict, column_order_list):
    """
    Converts a dictionary of binary masks into a pandas DataFrame,
    respecting a specified column order.

    Args:
        mask_dict (dict): A dictionary where keys are gene pair strings and
                          values are 1D numpy arrays (the masks).
        column_order_list (list): A list of gene pair strings specifying the
                                  desired order of the DataFrame columns.

    Returns:
        pd.DataFrame: A DataFrame with masks as columns, in the specified order.
    """
    # 1. Create a dictionary with only the ordered columns
    ordered_data = {col: mask_dict[col] for col in column_order_list if col in mask_dict}
    
    # 2. Check if all specified columns were found
    if len(ordered_data) != len(column_order_list):
        missing_columns = set(column_order_list) - set(ordered_data.keys())
        print(f"Warning: The following columns were not found in the mask dictionary: {missing_columns}")

    # 3. Create the DataFrame from the ordered dictionary
    df = pd.DataFrame(ordered_data)
    
    return df

def binarize_adata(adata, expression_threshold = 0):

    if issparse(adata.X):
        binary_expression = (adata.X > expression_threshold).astype(int).tocsr()
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
    mask = create_pairwise_binary_mask(ne, adata.var.index)
    mask = dict_to_dataframe(mask, column_order_list = grn_adata.var.index)
    grn_adata.layers['mask'] = mask
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