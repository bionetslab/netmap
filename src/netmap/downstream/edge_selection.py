import numpy as np
import pandas as pd
import numpy as np
from collections import Counter
import numpy as np
from scipy.sparse import issparse

def chunked_argsort(adata, layer_name='sorted', chunk_size=500, dtype=None):
    """
    Computes np.argsort on adata.X in chunks to save memory.
    
    Parameters:
    -----------
    adata : AnnData
        The AnnData object to process.
    layer_name : str
        The name of the layer where results will be stored.
    chunk_size : int
        Number of rows (cells) to process per iteration.
    dtype : np.dtype
        The integer type for the output. If None, it will automatically 
        choose uint16 or uint32 based on the number of genes.
    """
    n_obs, n_vars = adata.shape
    
    # 1. Automatically determine the smallest safe integer type
    if dtype is None:
        if n_vars < 65535:
            dtype = np.uint16
        else:
            dtype = np.uint32
            
    # 2. Pre-allocate the layer
    adata.layers[layer_name] = np.empty((n_obs, n_vars), dtype=dtype)
    
    # 3. Loop through chunks
    for i in range(0, n_obs, chunk_size):
        end = min(i + chunk_size, n_obs)
        
        # Pull chunk and densify only if necessary
        chunk = adata.X[i:end]
        if issparse(chunk):
            chunk = chunk.toarray()
            
        # Perform sort and assign
        adata.layers[layer_name][i:end] = np.argsort(chunk, axis=1)
        
    print(f"Successfully created layer '{layer_name}' using {dtype}.")



def _get_top_edges_global(grn_adata, top_edges: float):
    """
    Subroutine to get the top edges from an Anndata GRN object

    Parameters
    ----------
    grn_adata : AnnData
        A GRN anndata object to process containing a layer 'sorted'
        with the result of an argsort

    top_edges: float
        Percentage of interest

    Returns
    -------
    final_df : pd.DataFrame
        Processed Anndata object with the counted edges
    """
    if  not 'sorted' in grn_adata.layers:
        try:
            chunked_argsort(grn_adata)
        except np._core._exceptions._ArrayMemoryError:
            print(f"You ran into an issue sorting the array. Please manually sort"
                "the array using chunked_argsort and reduce the chunk size (current default chunk"
                " size: 500)")

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
        t_val = top_edges[i]

        top_edges_metadata = edge_metadata_np[top_idx.ravel()]
        edge_counts_map = Counter(top_edges_metadata.tolist())

        #top.append(get_top_edges_per_cell(grn_adata, top_idx, t_val))
        top.append(edge_counts_map)


    global_counter = top[0]
    final_df = [_data_preprr(global_counter, edge_metadata_np, top_edges[0])]
    for i in range(1, len(top)):
        global_counter = global_counter + top[i]
        t_val = top_edges[i]
        final_df.append(_data_preprr(global_counter, edge_metadata_np, t_val))
    
    final_df = np.concatenate(final_df)
    final_df = pd.DataFrame(final_df)
    return final_df

def _data_preprr(global_counter, edge_metadata_np, top_edges_val) -> pd.DataFrame:
    
    """
    Subroutine of to identify the top scoring edges and count how many times an edge
    is among the top edges.

    Parameters
    ----------
    global_counter : Counter
        A counter dictionary containing the keys of the edges

    edge_metadata_np: np.ndarray
        The names of the edges

    top_edges_valu: float 
        Top percentage to be returned with the data

    Returns
    -------
    final_summary_result : pd.DataFrame
        DataFrame with the edge counts per percentage
    """
    
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
    """
    Add annotation  colum(s) to the the var slot of the anndata object indicating whether the edge in
    in the top n% edges (by value) by cluster. A list of possible values can be passed, and should be, if multiple
    values are required, because the function requires sorting the object.

    Parameters
    ----------
    grn_adata : AnnData
        A GRN anndata object to process

    top_edges: list
        Percentages of interest

    nan_fill: int [Default: 0]
        Value to add when no edge is found
    
    key_name: str [Default: global]
        Prefix name of column(s) created

    Returns
    -------
    grn_adata : AnnData
        Processed Anndata object with the column(s) added
    """
        
    var = grn_adata.var
    if var.index.name is None or var.index.name == 'index':
        var = var.reset_index()
        var = var.rename(columns = {'index':'edge_key'})
    else:
        var = var.reset_index()
        
    for clu in grn_adata.obs[cluster_var].unique():
        grn_adata_sub = grn_adata[grn_adata.obs[cluster_var] == clu]
        top_edges_per_cell = _get_top_edges_global(grn_adata_sub,  top_edges)

        for te in top_edges:
            if f'cell_count_{te}_{clu}' in var.columns:
                continue
            var = var.merge(top_edges_per_cell.loc[top_edges_per_cell.top_edges==te, ['edge_key', 'cell_count']].rename(columns = {'cell_count': f'cell_count_{te}_{clu}'}), left_on = 'edge_key', right_on='edge_key', how='outer')
            var[f'cell_count_{te}_{clu}']= var[f'cell_count_{te}_{clu}'].fillna(nan_fill)

    var = var.set_index('edge_key')
    grn_adata.var = var

    return grn_adata


def add_top_edge_annotation_global(grn_adata, top_edges = [0.1], nan_fill = 0, key_name = 'global'):
    """
    Add annotation  colum(s) to the the var slot of the anndata object indicating whether the edge in
    in the top n% edges (by value). A list of possible values can be passed, and should be, if multiple
    values are required, because the function requires sorting the object.

    Parameters
    ----------
    grn_adata : AnnData
        A GRN anndata object to process

    top_edges: list
        Percentages of interest

    nan_fill: int [Default: 0]
        Value to add when no edge is found
    
    key_name: str [Default: global]
        Prefix name of column(s) created

    Returns
    -------
    grn_adata : AnnData
        Processed Anndata object with the column(s) added
    """

    var = grn_adata.var
    if var.index.name is None or var.index.name == 'index':
        var = var.reset_index()
        var = var.rename(columns = {'index':'edge_key'})
    else:
        var = var.reset_index()
        
    top_edges_per_cell = _get_top_edges_global(grn_adata,  top_edges)
    for te in top_edges:
        if f'{key_name}_cell_count_{te}' in var.columns:
            continue
        var = var.merge(top_edges_per_cell.loc[top_edges_per_cell.top_edges==te, ['edge_key', 'cell_count']].rename(columns = {'cell_count': f'{key_name}_cell_count_{te}'}), left_on = 'edge_key', right_on='edge_key', how='outer')
        var[f'{key_name}_cell_count_{te}']= var[f'{key_name}_cell_count_{te}'].fillna(nan_fill)

    var = var.set_index('edge_key')
    grn_adata.var = var
    
    return grn_adata