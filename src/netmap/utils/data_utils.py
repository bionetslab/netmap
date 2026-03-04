import anndata
import os
import os.path as op

import pandas as pd
import scipy.sparse
import h5py

from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import anndata as ad

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


def retrieve_top_edges(grn_adata, output_dir, percentage=0.1):

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    parquet_files = list(output_dir.glob("*.parquet"))

    grn_adata.uns['edge_sum_original_index'] = grn_adata.var['edge_sums'].values.argsort()[::-1]

    # Slice the object
    max_entry = grn_adata.var.shape[0]
    max_index = int(max_entry*percentage)
    sorted_indices = np.sort(grn_adata.uns['edge_sum_original_index'][0:max_index])
    index_set = set(grn_adata.var['index'][sorted_indices])

    # 2. Map columns to files (fast, reads only metadata)
    col_to_file = {}
    for f in parquet_files:
        meta = pq.read_metadata(f)
        for i in range(meta.num_columns):
            col_name = meta.schema.names[i]
            if col_name in index_set:
                if col_name not in col_to_file:
                    col_to_file[col_name] = []
                col_to_file[col_name].append(f)

    file_to_req_cols = {}
    for col, files in col_to_file.items():
        f = files[0] # Assuming column names are unique across files
        file_to_req_cols.setdefault(f, []).append(col)

    sub_tables = []
    for f, cols in file_to_req_cols.items():
        # This only pulls the specific columns from disk
        sub_tables.append(pq.read_table(f, columns=cols))

    full_table = pa.Table.from_arrays(
        [col for t in sub_tables for col in t.columns],
        names=[name for t in sub_tables for name in t.column_names]
    )

    grn_adata = grn_adata[:, sorted_indices]
    grn_adata = ad.AnnData(full_table.to_pandas().to_numpy(), var = grn_adata.var, obs = grn_adata.obs)
    
    return grn_adata


def retrieve_edges_by_index(grn_adata, output_dir, index_list):

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    parquet_files = list(output_dir.glob("*.parquet"))
    sorted_indices = np.sort(index_list)

    index_set = set(grn_adata.var.index[sorted_indices])

    # 2. Map columns to files (fast, reads only metadata)
    col_to_file = {}
    for f in parquet_files:
        meta = pq.read_metadata(f)
        for i in range(meta.num_columns):
            col_name = meta.schema.names[i]
            if col_name in index_set:
                if col_name not in col_to_file:
                    col_to_file[col_name] = []
                col_to_file[col_name].append(f)

    file_to_req_cols = {}
    for col, files in col_to_file.items():
        f = files[0] # Assuming column names are unique across files
        file_to_req_cols.setdefault(f, []).append(col)

    sub_tables = []
    for f, cols in file_to_req_cols.items():
        # This only pulls the specific columns from disk
        sub_tables.append(pq.read_table(f, columns=cols))

    full_table = pa.Table.from_arrays(
        [col for t in sub_tables for col in t.columns],
        names=[name for t in sub_tables for name in t.column_names]
    )

    grn_adata = grn_adata[:, sorted_indices]
    grn_adata = ad.AnnData(full_table.to_pandas().to_numpy(), var = grn_adata.var, obs = grn_adata.obs)
    
    return grn_adata


def remove_regulon_columns(grn_adata):
    
    """
    Utility function to remove regulon columns, if for some reason you added it
    and need to remove it when scanpy plotting starts complaining.

    """
    spike_cols = [col for col in grn_adata.obs.columns if 'regulon' in col]
    grn_adata.obs = grn_adata.obs.drop(columns = spike_cols)
    return grn_adata

def remove_Ucell_columns(grn_adata):
    
    """
    Utility function to remove regulon columns, if for some reason you added it
    and need to remove it when scanpy plotting starts complaining.

    """
    spike_cols = [col for col in grn_adata.obs.columns if 'UCell' in col]
    grn_adata.obs = grn_adata.obs.drop(columns = spike_cols)
    return grn_adata