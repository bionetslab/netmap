import anndata
import os
import os.path as op

import pandas as pd
import scipy.sparse
import h5py

def attribution_to_anndata(attribution_list, var = None, obs = None)-> anndata.AnnData:

    """
    Transform attribution data frame into an anndata object

    Args:
        attribution_list: (sparse) Data frame of attribution values (one column per edge)

    returns: 
        anndata.Anndata: Anndata object with attribution values in X.
    """
    print('Creating anndata')
    adata = anndata.AnnData(attribution_list)
    #adata.raw = adata
    if var is not None:
        print('Setting vars')
        adata.var = var
    if obs is not None:
        adata.obs = obs
    return adata


def create_output_directory(result_params):
    os.makedirs(result_params['output_directory'], exist_ok=result_params['overwrite'])


def save_anndata(adobj, result_params):
    adobj.write( filename = op.join(result_params['output_directory'], result_params['adata_filename']))



def merge_all_to_obs(target_adata, source_adata, replace=True):
    """
    Takes all variables from source_adata and appends them as columns
    to target_adata.obs for easy plotting.
    """
    if target_adata.n_obs != source_adata.n_obs:
        raise ValueError("Cell counts do not match between objects.")


    if scipy.sparse.issparse(source_adata.X):
        source_data = source_adata.X.toarray()
    else:
        source_data = source_adata.X

    # Create a DataFrame from the source data
    source_df = pd.DataFrame(
        source_data, 
        index=source_adata.obs_names, 
        columns=source_adata.var_names
    )

    # Check if regulon cols are already present, and delte all regulon columns
    if len(set(target_adata.obs.columns).intersection(list(source_df.columns)))>0:
        if replace:
            spike_cols = [col for col in target_adata.obs.columns if 'regulon' in col]
            target_adata.obs = target_adata.obs.drop(columns = spike_cols)
            target_adata.obs = pd.concat([target_adata.obs, source_df], axis=1)
        else:
            print('Regulon columns where present and not replaced.')
    else:
        target_adata.obs = pd.concat([target_adata.obs, source_df], axis=1)


    return target_adata


def retrieve_top_edges(grn_adata, percentage = 0.1):

    grn_adata.uns['edge_sum_original_index'] = grn_adata.var['edge_sums'].values.argsort()[::-1]
    
    # Slice the object
    max_entry = grn_adata.var.shape[0]
    max_index = int(max_entry*percentage)
    sorted_indices = np.sort(grn_adata.uns['edge_sum_original_index'])
    sorted_indices = sorted_indices[0:max_index]

    with h5py.File(grn_adata.uns['backing_file'], 'r+') as f:
        dset = f['data']

        X = dset[:, sorted_indices]
        new_grn_adata = ad.AnnData(X = X, var = grn_adata.var.iloc[sorted_indices], obs = grn_adata.obs, uns = grn_adata.uns)
        new_grn_adata.uns['complete_grn'] = False
    return new_grn_adata


def retrieve_edges_by_column(grn_adata, edges_keys, column_name = 'index'):
    
    indices = list(var[var[column_name].isin(edges_keys)].index)
    with h5py.File(grn_adata.uns['backing_file'], 'r+') as f:
        dset = f['data']
        X = dset[:, indices]
        new_grn_adata = ad.AnnData(X = X, var = grn_adata.var.iloc[indices], obs = grn_adata.obs, uns = grn_adata.uns)
        new_grn_adata.uns['complete_grn'] = False
    return new_grn_adata


def remove_regulon_columns(grn_adata):
    
    """
    Utility function to remove regulon columns, if for some reason you added it
    and need to remove it when scanpy plotting starts complaining.

    """
    spike_cols = [col for col in grn_adata.obs.columns if 'regulon' in col]
    grn_adata.obs = grn_adata.obs.drop(columns = spike_cols)
    return grn_adata