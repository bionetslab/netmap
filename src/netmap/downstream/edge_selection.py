"""Memory-efficient edge ranking and top-edge annotation for GRN AnnData objects.

:func:`add_top_edge_annotation_global` and :func:`add_top_edge_annotation_cluster`
annotate ``grn_adata.var`` with per-cell counts of how often each edge appears
in the top-N% of attributions, globally or per Leiden cluster respectively.
"""

import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import issparse


def chunked_argsort(unsadata, layer_name='sorted', chunk_size=500, dtype=None):
    """Compute ``np.argsort`` on ``adata.X`` in chunks to reduce peak memory usage.

    Processes ``unsadata.X`` row-by-row in batches of ``chunk_size``, writing
    the sorted column-index order for each row into a pre-allocated layer.
    The integer dtype is chosen automatically to be as small as possible given
    the number of variables.

    Args:
        unsadata (anndata.AnnData): AnnData object to process.  Modified
            in-place by adding ``layers[layer_name]``.
        layer_name (str): Name of the new layer to create.  Defaults to
            ``'sorted'``.
        chunk_size (int): Number of rows (cells) to process per iteration.
            Defaults to 500.
        dtype (numpy.dtype or None): Integer dtype for the output array.
            If ``None``, ``uint16`` is used when ``n_vars < 65535``, otherwise
            ``uint32``.

    Returns:
        None. Modifies ``unsadata`` in-place by adding
        ``layers[layer_name]``.
    """
    n_obs, n_vars = unsadata.shape

    # 1. Automatically determine the smallest safe integer type
    if dtype is None:
        if n_vars < 65535:
            dtype = np.uint16
        else:
            dtype = np.uint32

    # 2. Pre-allocate the layer
    unsadata.layers[layer_name] = np.empty((n_obs, n_vars), dtype=dtype)

    # 3. Loop through chunks
    for i in range(0, n_obs, chunk_size):
        end = min(i + chunk_size, n_obs)

        # Pull chunk and densify only if necessary
        chunk = unsadata.X[i:end]
        if issparse(chunk):
            chunk = chunk.toarray()

        # Perform sort and assign
        unsadata.layers[layer_name][i:end] = np.argsort(chunk, axis=1)

    print(f"Successfully created layer '{layer_name}' using {dtype}.")


def _get_top_edges_global(grn_adata, top_edges: float):
    """Return edge-count DataFrames for multiple top-percentage thresholds.

    Uses the pre-sorted ``'sorted'`` layer (created by :func:`chunked_argsort`)
    to count, for each threshold in ``top_edges``, how many cells include each
    edge among their top-N% highest-attribution edges.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData object containing a layer
            ``'sorted'`` with the result of an argsort (column indices ordered
            from lowest to highest attribution per row).
        top_edges (list of float): Fractions of edges to consider as "top"
            (e.g. ``[0.05, 0.1]``).

    Returns:
        pandas.DataFrame: DataFrame with columns ``'edge_key'``,
            ``'top_edges'``, and ``'cell_count'``, covering all thresholds
            in ``top_edges`` concatenated into a single frame.
    """
    #if  not 'sorted' in grn_adata.layers:
    try:
        chunked_argsort(grn_adata)
    except np._core._exceptions._ArrayMemoryError:
        print(f"You ran into an issue sorting the array. Please manually sort"
            "the array using chunked_argsort and reduce the chunk size (current default chunk"
            " size: 500)")
    except MemoryError:
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
    """Build a structured array recording per-edge cell counts at one threshold.

    Args:
        global_counter (collections.Counter): Mapping of edge name → number of
            cells for which the edge appeared in the top-percentage slice.
        edge_metadata_np (numpy.ndarray): Full array of edge names from
            ``grn_adata.var.index``, used to infer the correct string dtype.
        top_edges_val (float): The top-percentage threshold value to embed in
            the ``'top_edges'`` field of the output.

    Returns:
        numpy.ndarray: Structured array with fields ``'edge_key'`` (edge name),
            ``'top_edges'`` (float16 threshold), and ``'cell_count'`` (int32
            number of cells).
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


def add_top_edge_annotation_cluster(grn_adata, top_edges=[0.1], nan_fill=0, cluster_var='spectral'):
    """Annotate ``grn_adata.var`` with per-cluster top-edge cell counts.

    For each cluster in ``grn_adata.obs[cluster_var]``, counts how many cells
    within that cluster include each edge among their top-N% highest-attribution
    edges.  Results are merged into ``grn_adata.var`` as new columns.

    Pass multiple values in ``top_edges`` when several thresholds are needed;
    sorting is performed only once regardless of how many thresholds are
    supplied.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData object to process.
        top_edges (list of float): Fractions of top edges to count.
            Defaults to ``[0.1]``.
        nan_fill (int or float): Value used to fill edges with no count at a
            given threshold/cluster combination.  Defaults to 0.
        cluster_var (str): Column in ``grn_adata.obs`` that identifies cluster
            membership.  Defaults to ``'spectral'``.

    Returns:
        anndata.AnnData: ``grn_adata`` with new columns added to ``.var``
            named ``'cell_count_{top_edge}_{cluster}'`` for every combination
            of threshold and cluster.
    """
    var = grn_adata.var
    if var.index.name is None or var.index.name == 'index':
        var = var.reset_index()
        var = var.rename(columns={'index': 'edge_key'})
    else:
        var = var.reset_index()

    for clu in grn_adata.obs[cluster_var].unique():
        grn_adata_sub = grn_adata[grn_adata.obs[cluster_var] == clu]
        top_edges_per_cell = _get_top_edges_global(grn_adata_sub, top_edges)

        for te in top_edges:
            if f'cell_count_{te}_{clu}' in var.columns:
                continue
            var = var.merge(top_edges_per_cell.loc[top_edges_per_cell.top_edges == te, ['edge_key', 'cell_count']].rename(columns={'cell_count': f'cell_count_{te}_{clu}'}), left_on='edge_key', right_on='edge_key', how='outer')
            var[f'cell_count_{te}_{clu}'] = var[f'cell_count_{te}_{clu}'].fillna(nan_fill)

    var = var.set_index('edge_key')
    grn_adata.var = var

    return grn_adata


def add_top_edge_annotation_global(grn_adata, top_edges=[0.1], nan_fill=0, key_name='global'):
    """Annotate ``grn_adata.var`` with global top-edge cell counts.

    Counts, across all cells, how many times each edge appears among the
    top-N% highest-attribution edges.  Results are merged into
    ``grn_adata.var`` as new columns.

    Pass multiple values in ``top_edges`` when several thresholds are needed;
    sorting is performed only once regardless of how many thresholds are
    supplied.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData object to process.
        top_edges (list of float): Fractions of top edges to count.
            Defaults to ``[0.1]``.
        nan_fill (int or float): Value used to fill edges with no count at a
            given threshold.  Defaults to 0.
        key_name (str): Prefix for the new column names.  Defaults to
            ``'global'``.

    Returns:
        anndata.AnnData: ``grn_adata`` with new columns added to ``.var``
            named ``'{key_name}_cell_count_{top_edge}'`` for each threshold.
    """
    var = grn_adata.var
    if var.index.name is None or var.index.name == 'index':
        var = var.reset_index()
        var = var.rename(columns={'index': 'edge_key'})
    else:
        var = var.reset_index()

    top_edges_per_cell = _get_top_edges_global(grn_adata, top_edges)
    for te in top_edges:
        if f'{key_name}_cell_count_{te}' in var.columns:
            continue
        var = var.merge(top_edges_per_cell.loc[top_edges_per_cell.top_edges == te, ['edge_key', 'cell_count']].rename(columns={'cell_count': f'{key_name}_cell_count_{te}'}), left_on='edge_key', right_on='edge_key', how='outer')
        var[f'{key_name}_cell_count_{te}'] = var[f'{key_name}_cell_count_{te}'].fillna(nan_fill)

    var = var.set_index('edge_key')
    grn_adata.var = var

    return grn_adata
