"""I/O utilities for AnnData objects and Parquet-backed GRN edge data.

:func:`retrieve_top_edges` and :func:`retrieve_edges_by_index` load specific
attribution columns from the Parquet shards written by
:func:`~netmap.grn.inferrence.inferrence` into ``grn_adata.X``.
"""

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

def attribution_to_anndata(attribution_list, var=None, obs=None) -> anndata.AnnData:
    """Transform an attribution data frame into an AnnData object.

    Args:
        attribution_list: (Sparse) data frame of attribution values, with one
            column per directed edge (SourceGene_TargetGene).
        var: Optional DataFrame to assign as ``adata.var`` (edge metadata,
            e.g. source, target, edge_sums). Must have the same number of rows
            as columns in ``attribution_list``.
        obs: Optional DataFrame to assign as ``adata.obs`` (cell metadata).
            Must have the same number of rows as ``attribution_list``.

    Returns:
        anndata.AnnData: AnnData object with attribution values stored in ``X``,
            and optional ``var`` / ``obs`` metadata attached.
    """
    print('Creating anndata')
    adata = anndata.AnnData(attribution_list)
    if var is not None:
        print('Setting vars')
        adata.var = var
    if obs is not None:
        adata.obs = obs
    return adata


def create_output_directory(result_params):
    """Create the output directory defined in result_params.

    Args:
        result_params (dict): Must contain ``'output_directory'`` (str) and
            ``'overwrite'`` (bool, passed as ``exist_ok``).
    """
    os.makedirs(result_params['output_directory'], exist_ok=result_params['overwrite'])


def save_anndata(adobj, result_params):
    """Save an AnnData object to disk.

    Args:
        adobj (anndata.AnnData): Object to save.
        result_params (dict): Must contain ``'output_directory'`` and
            ``'adata_filename'`` keys.
    """
    adobj.write(filename=op.join(result_params['output_directory'], result_params['adata_filename']))


def merge_all_to_obs(target_adata, source_adata, replace=True):
    """Copies all variables from source_adata into target_adata.obs for convenient Scanpy plotting.

    All columns in ``source_adata.var_names`` are appended as new columns to
    ``target_adata.obs``.  When columns named ``'regulon'`` are already present
    in ``target_adata.obs`` and ``replace=True``, those existing regulon columns
    are dropped before the new values are concatenated.

    Args:
        target_adata: AnnData object whose ``obs`` DataFrame will be extended.
            Must have the same number of cells as ``source_adata``.
        source_adata: AnnData object whose ``X`` matrix (cells x variables) is
            used as the source of new ``obs`` columns.
        replace: If ``True`` (default), any existing columns whose name contains
            ``'regulon'`` in ``target_adata.obs`` are removed before adding the
            new columns.  If ``False``, the merge is skipped when overlapping
            column names are detected and a message is printed.

    Returns:
        target_adata: The input ``target_adata`` with updated ``obs`` columns.

    Raises:
        ValueError: If the cell counts of ``target_adata`` and ``source_adata``
            do not match.
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

    # Check if regulon cols are already present, and delete all regulon columns
    if len(set(target_adata.obs.columns).intersection(list(source_df.columns))) > 0:
        if replace:
            spike_cols = [col for col in target_adata.obs.columns if 'regulon' in col]
            target_adata.obs = target_adata.obs.drop(columns=spike_cols)
            target_adata.obs = pd.concat([target_adata.obs, source_df], axis=1)
        else:
            print('Regulon columns where present and not replaced.')
    else:
        target_adata.obs = pd.concat([target_adata.obs, source_df], axis=1)

    return target_adata


def retrieve_top_edges(grn_adata, output_dir, percentage=0.1, inplace=True):
    """Load the top-percentage edges by attribution sum from Parquet shards.

    Selects the top ``percentage`` fraction of edges ranked by ``edge_sums``,
    then reads the corresponding columns from the Parquet shards in ``output_dir``.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData with empty ``X`` (Parquet-backed).
        output_dir (str or Path): Directory containing ``*.parquet`` shard files.
        percentage (float): Fraction of top edges to load. Defaults to 0.1.
        inplace (bool): If ``True``, slice ``grn_adata`` in-place and set ``X``.
            If ``False``, create a new AnnData. Defaults to ``True``.

    Returns:
        anndata.AnnData: GRN AnnData with ``X`` populated for the selected edges.
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    parquet_files = list(output_dir.glob("*.parquet"))

    grn_adata.uns['edge_sum_original_index'] = grn_adata.var['edge_sums'].values.argsort()[::-1]

    # Slice the object
    max_entry = grn_adata.var.shape[0]
    max_index = int(max_entry * percentage)
    sorted_indices = np.sort(grn_adata.uns['edge_sum_original_index'][0:max_index])
    index_set = set(grn_adata.var['index'].iloc[sorted_indices])

    # Map columns to files (fast, reads only metadata)
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
        f = files[0]  # Assuming column names are unique across files
        file_to_req_cols.setdefault(f, []).append(col)

    sub_tables = []
    for f, cols in file_to_req_cols.items():
        # This only pulls the specific columns from disk
        sub_tables.append(pq.read_table(f, columns=cols))

    full_table = pa.Table.from_arrays(
        [col for t in sub_tables for col in t.columns],
        names=[name for t in sub_tables for name in t.column_names]
    )

    if not inplace:
        grn_adata = grn_adata[:, sorted_indices]
        grn_adata = ad.AnnData(full_table.to_pandas().to_numpy(), var=grn_adata.var, obs=grn_adata.obs)
    else:
        grn_adata = grn_adata[:, sorted_indices].copy()
        grn_adata.X = full_table.to_pandas().to_numpy()
    return grn_adata


def retrieve_edges_by_index(grn_adata, output_dir, index_list, inplace=True):
    """Load specific edges by integer index from Parquet shards.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData with empty ``X`` (Parquet-backed).
        output_dir (str or Path): Directory containing ``*.parquet`` shard files.
        index_list (list of int): Integer indices into ``grn_adata.var`` to load.
        inplace (bool): If ``True``, slice and update in-place. Defaults to ``True``.

    Returns:
        anndata.AnnData: GRN AnnData with ``X`` populated for the selected edges.
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    parquet_files = list(output_dir.glob("*.parquet"))
    sorted_indices = np.sort(index_list)

    index_set = set(grn_adata.var.index[sorted_indices])

    # Map columns to files (fast, reads only metadata)
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
        f = files[0]  # Assuming column names are unique across files
        file_to_req_cols.setdefault(f, []).append(col)

    sub_tables = []
    for f, cols in file_to_req_cols.items():
        # This only pulls the specific columns from disk
        sub_tables.append(pq.read_table(f, columns=cols))

    full_table = pa.Table.from_arrays(
        [col for t in sub_tables for col in t.columns],
        names=[name for t in sub_tables for name in t.column_names]
    )

    if not inplace:
        grn_adata = grn_adata[:, sorted_indices]
        grn_adata = ad.AnnData(full_table.to_pandas().to_numpy(), var=grn_adata.var, obs=grn_adata.obs)
    else:
        grn_adata = grn_adata[:, sorted_indices].copy()
        grn_adata.X = full_table.to_pandas().to_numpy()
    return grn_adata


def remove_regulon_columns(grn_adata):
    """Removes all regulon columns from grn_adata.obs.

    Useful when regulon activity values were previously added via
    ``merge_all_to_obs`` and need to be stripped out before Scanpy plotting
    functions complain about unrecognised observation columns.

    Args:
        grn_adata: AnnData object whose ``obs`` DataFrame may contain columns
            with ``'regulon'`` in their name.

    Returns:
        grn_adata: The input object with regulon columns removed from ``obs``.
    """
    spike_cols = [col for col in grn_adata.obs.columns if 'regulon' in col]
    grn_adata.obs = grn_adata.obs.drop(columns=spike_cols)
    return grn_adata


def remove_Ucell_columns(grn_adata):
    """Removes all UCell columns from grn_adata.obs.

    Useful when UCell scores were previously added to ``obs`` and need to be
    stripped out before Scanpy plotting functions complain about unrecognised
    observation columns.

    Args:
        grn_adata: AnnData object whose ``obs`` DataFrame may contain columns
            with ``'UCell'`` in their name.

    Returns:
        grn_adata: The input object with UCell columns removed from ``obs``.
    """
    spike_cols = [col for col in grn_adata.obs.columns if 'UCell' in col]
    grn_adata.obs = grn_adata.obs.drop(columns=spike_cols)
    return grn_adata
