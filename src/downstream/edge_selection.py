import numpy as np
import pandas as pd
import numpy as np
from collections import Counter

def _get_top_edges_global(grn_adata, top_edges: int, layer = 'X'):

    b = grn_adata.layers['sorted']
    # Calculate partition indices for all top_edges values
    top_edges_data_list = [int(np.round(grn_adata.shape[1] * t)) for t in top_edges]
    partition_indices = [grn_adata.shape[1]]+[grn_adata.shape[1] - n for n in top_edges_data_list]
    
    top = []
    edge_metadata_np = grn_adata.var.index.to_numpy()

    for i in range(len(partition_indices)-1):
        
        # part index is running backwards
        end_idx = partition_indices[i]
        start_index = partition_indices[i+1]
        top_idx = b[:, start_index:end_idx]
        print(start_index)
        print(end_idx)
        
        t_val = top_edges[i]

        top_edges_metadata = edge_metadata_np[top_idx.ravel()]
        edge_counts_map = Counter(top_edges_metadata.tolist())

        #top.append(get_top_edges_per_cell(grn_adata, top_idx, t_val))
        top.append(edge_counts_map)


    global_counter = top[0]
    final_df = [data_preprr(global_counter, edge_metadata_np, top_edges[0])]
    for i in range(1, len(top)):
        global_counter = global_counter + top[i]
        t_val = top_edges[i]
        final_df.append(data_preprr(global_counter, edge_metadata_np, t_val))
    
    final_df = np.concatenate(final_df)
    final_df = pd.DataFrame(final_df)
    return final_df

def data_preprr(global_counter, edge_metadata_np, top_edges_val):
    
    edge_keys_list = []
    cell_counts_list = []

    # Iterating over items() is generally faster than two separate list comprehensions
    for key, count in global_counter.items():
        edge_keys_list.append(key)
        cell_counts_list.append(count)

    # Convert to NumPy arrays
    edge_keys_np = np.array(edge_keys_list, dtype=edge_metadata_np.dtype)
    cell_counts_np = np.array(cell_counts_list, dtype=np.int32)

    # Get the size of the result
    N = len(edge_keys_np)

    # Define dtype_final using the *full* index data type
    dtype_final = np.dtype([
        # Use the dtype of the original index, which is now the complete index
        ('edge_key', edge_metadata_np.dtype),
        ('top_edges', np.float16),
        ('cell_count', np.int32)
    ])

    # Create the empty structured array
    final_summary_result = np.empty(N, dtype=dtype_final)

    # Populate the structured array fields
    # The index is now final_edge_keys
    final_summary_result['edge_key'] = edge_keys_np
    # The counts array is now cell_counts_reindexed
    final_summary_result['cell_count'] = cell_counts_np
    final_summary_result['top_edges'] = top_edges_val

    return final_summary_result



def add_top_edge_annotation_cluster(grn_adata, top_edges = [0.1], nan_fill = 0, cluster_var = 'spectral'):
    var = grn_adata.var
    if var.index.name is None or var.index.name == 'index':
        var = var.reset_index()
        var = var.rename(columns = {'index':'edge_key'})
    else:
        var = var.reset_index()
        
    for clu in grn_adata.obs[cluster_var].unique():
        grn_adata_sub = grn_adata[grn_adata.obs[cluster_var] == clu]
        top_edges_per_cell = _get_top_edges_global(grn_adata_sub,  top_edges, layer='X')

        for te in top_edges:
            if f'cell_count_{te}_{clu}' in var.columns:
                continue
            var = var.merge(top_edges_per_cell.loc[top_edges_per_cell.top_edges==te, ['edge_key', 'cell_count']].rename(columns = {'cell_count': f'cell_count_{te}_{clu}'}), left_on = 'edge_key', right_on='edge_key', how='outer')
            var[f'cell_count_{te}_{clu}']= var[f'cell_count_{te}_{clu}'].fillna(nan_fill)

    var = var.set_index('edge_key')
    grn_adata.var = var

    return grn_adata


def add_top_edge_annotation_global(grn_adata, top_edges = [0.1], nan_fill = 0):

    var = grn_adata.var
    if var.index.name is None or var.index.name == 'index':
        var = var.reset_index()
        var = var.rename(columns = {'index':'edge_key'})
    else:
        var = var.reset_index()
        
    top_edges_per_cell = _get_top_edges_global(grn_adata,  top_edges, layer='X')
    for te in top_edges:
        if f'global_cell_count_{te}' in var.columns:
            continue
        var = var.merge(top_edges_per_cell.loc[top_edges_per_cell.top_edges==te, ['edge_key', 'cell_count']].rename(columns = {'cell_count': f'global_cell_count_{te}'}), left_on = 'edge_key', right_on='edge_key', how='outer')
        var[f'global_cell_count_{te}']= var[f'global_cell_count_{te}'].fillna(nan_fill)

    var = var.set_index('edge_key')
    grn_adata.var = var
    
    return grn_adata