import pandas as pd
import numpy as np

def _create_edge_mask_from_GRN(grn_df, gene_list, name_grn='external_grn'):
    """
    Create flat vector mask for TF target interactions based on literature GRN and 
    gene list.
    
    Args:
        grn_df (pd.DataFrame): A DataFrame representing the GRN with columns 'source',
                               'target', and 'weight'.
        gene_list (list): A list of gene names to be included in the matrix. (should be the gene in the study)

    Returns:
        flat_vector: numpy.ndarray: numeric mask containing the value of the GRN
        names: numpy.ndarray: edge name vector (GeneA_GeneB)

    """
    # Create a mapping from gene names to their matrix indices for efficient look-up.
    gene_to_index = {gene: i for i, gene in enumerate(gene_list)}
    num_genes = len(gene_list)

    matrix = np.zeros((num_genes, num_genes))

    # Iterate through each row of the GRN DataFrame and populate the matrix.
    for _, row in grn_df.iterrows():
        tf = row['source']
        target = row['target']
        weight = row['weight']

        # Check if both the TF and target are in our target gene list.
        if tf in gene_to_index and target in gene_to_index:
            tf_index = gene_to_index[tf]
            target_index = gene_to_index[target]
            
            # Populate the matrix with the corresponding weight.
            matrix[tf_index, target_index] = weight

    # Create edge names
    edge_names = []
    for gene_A in gene_list:
        for gene_B in gene_list:
            edge_names.append(f'{gene_A}_{gene_B}')

    edge_mask = matrix.flatten()
    edge_mask = pd.DataFrame({'index': edge_names, f'edge_in_{name_grn}': edge_mask})
    edge_mask =edge_mask.set_index('index')
    
    return edge_mask



def _get_all_genes_in_grn_object(grnad):
    all_sources = np.unique(grnad.var.source)
    all_targets = np.unique(grnad.var.target)
    all_genes = np.unique(np.concatenate([all_sources, all_targets]))
    return all_genes


def add_external_grn(grn_ad, external_grn, name_grn = 'external_grn'):
    
    """
    Adds three columns to a anndate GRN object. 
    is_target
    is_source
    is_egde
    
    """

    all_my_genes = _get_all_genes_in_grn_object(grn_ad)
    edge_mask = _create_edge_mask_from_GRN(external_grn, all_my_genes, name_grn = name_grn)
    grn_ad.var = grn_ad.var.merge(edge_mask, left_index=True, right_index=True)
    grn_ad.var[f'is_target_{name_grn}'] = grn_ad.var.target.isin(external_grn.target)
    grn_ad.var[f'is_source_{name_grn}'] = grn_ad.var.source.isin(external_grn.source)
    return grn_ad

