import numpy as np
import pandas as pd

def _get_top_edges_global(grn_adata, top_edges: int, layer = 'X'):
    # Select the correct data based on the layer
    if layer == 'X':
        data = grn_adata.X
    else:
        data = grn_adata.layers[layer]
     
    b = np.argsort(data, axis=1)
    # Calculate partition indices for all top_edges values
    top_edges_data_list = [int(np.round(grn_adata.shape[1] * t)) for t in top_edges]
    partition_indices = [grn_adata.shape[1] - n for n in top_edges_data_list]
    
    print(top_edges)
    top = []
    
    for t_val, part_idx in zip(top_edges, partition_indices):
        # Subset the argsorted array *once* for each t_val
        top_idx = b[:, part_idx:]
        
        top.append(_get_top_edges_per_cell(grn_adata, top_idx, t_val))

    top = pd.concat(top, ignore_index=True)
    return top


def _get_top_edges_per_cell(grn_adata, top_idx_for_t, top_edges_val):
    """
    Calculates the number of cells that contain each unique top edge using 
    NumPy's vectorized unique counting, which is faster than Pandas groupby 
    on Python objects (tuples).
    """
    
    # 1. Gene pair metadata is constant
    # Ensure this is a NumPy array for fast indexing
    edge_metadata_np = grn_adata.var.index.to_numpy()
  
    top_edges_metadata = edge_metadata_np[top_idx_for_t.ravel()]


    # 4. Count unique rows (edges) in the structured array
    unique_edges_structured, cell_counts = np.unique(
        top_edges_metadata,
        return_counts=True
    )
    
    # 5. Extract results and create the DataFrame directly
    summary_df = pd.DataFrame({
        'edge_key': unique_edges_structured,
        'cell_count': cell_counts
    })
    
    # 6. Add the constant metadata
    summary_df['top_edges'] = top_edges_val
    
    # 7. Select and reorder final columns:
    final_cols = ['edge_key', 'top_edges', 'cell_count']


    return summary_df[final_cols]



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