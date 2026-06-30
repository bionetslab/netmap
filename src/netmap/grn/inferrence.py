"""GRN inference via Captum attributions over an autoencoder ensemble.

The main entry point :func:`inferrence` iterates over all target genes,
applies the XAI method across all models in the zoo, aggregates attributions,
and writes one Parquet shard per gene to ``<output_dir>/grn/``.  The returned
:class:`~anndata.AnnData` object (``grn_adata``) has cells as observations and
directed edges (``SourceGene_TargetGene``) as variables; ``X`` is an empty
placeholder backed by those Parquet files until loaded with
:func:`~netmap.utils.data_utils.retrieve_top_edges`.
"""

import numpy as np
from captum.attr import GuidedBackprop, GradientShap, Deconvolution
#import pingouin as pingu
import torch
from tqdm import tqdm

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from netmap.utils.data_utils import attribution_to_anndata
import itertools
from scipy import integrate
import scipy.stats
import h5py
import anndata as ad

import os
import os.path as op


def _quantile_partitioning(data: np.ndarray, q: int) -> np.ndarray:
    """
    Performs quantile partitioning on a 1D NumPy array.

    The method orders the data and divides it into 'q' equal-sized partitions.
    A new array (mask) is created where each element is assigned a value
    of k/q, where k is the quantile that the element belongs to.

    Args:
        data (np.ndarray): A 1D NumPy array of numerical data.
        q (int): The number of quantiles to partition the data into. Must be a
                 positive integer.

    Returns:
        np.ndarray: A new array of the same shape as the input data, with
                    values representing the quantile partition.

    Raises:
        ValueError: If q is not a positive integer or if the input data is not
                    a 1D NumPy array.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input data must be a 1D NumPy array.")
    if not isinstance(q, int) or q <= 0:
        raise ValueError("The number of quantiles 'q' must be a positive integer.")

    n = len(data)
    if n == 0:
        return np.array([])

    # 1. Order data using argsort to get the indices that would sort the array
    sorted_indices = np.argsort(data)

    # 2. Cut into q equal pieces by calculating the size of each partition
    # We use float division to handle cases where n is not perfectly divisible by q.
    partition_size = n / q

    # 3. Initialize mask with same dimension as data
    mask = np.zeros_like(data, dtype=float)

    # 4. With k=current quantile add k/q to all cells belonging to k
    for k in range(1, q + 1):
        start_index = int((k - 1) * partition_size)
        end_index = int(k * partition_size)

        # Get the original indices that belong to the current quantile
        quantile_indices = sorted_indices[start_index:end_index]

        # Assign the value k/q to the corresponding positions in the mask
        mask[quantile_indices] = k / q

    return mask

def _quantile_partitioning_2d(data: np.ndarray, q: int) -> np.ndarray:
    """
    Performs quantile partitioning row-wise on a 2D NumPy array using
    np.apply_along_axis for efficiency.

    Args:
        data (np.ndarray): A 2D NumPy array of numerical data.
        q (int): The number of quantiles to partition each row into.

    Returns:
        np.ndarray: A new 2D array with the same shape as the input,
                    where each row contains the quantile partitions.

    Raises:
        ValueError: If the input data is not a 2D NumPy array.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")

    # Use np.apply_along_axis to apply the 1D function to each row (axis=1).
    return np.apply_along_axis(_quantile_partitioning, 1, data, q)

# def compute_correlation_metric(data, cor_type):
#     # Compute gene correlation measure
#     #  'pingouin.pcorr', 'np.cov', 'np.corcoeff'
#     if cor_type ==  'pingouin.pcorr':
#         cov = pingu.pcorr(pd.DataFrame(data))
#     elif cor_type == 'np.cov':
#         cov = np.cov(data.T)
#     elif cor_type == 'np.corrcoeff':
#         cov = np.corrcoef(data.T)
#     elif cor_type == 'None':
#         cov = 1
#     else:
#         cov = 1
#     return cov

def aggregate_attributions(attributions, strategy='mean'):
    """Aggregate a list of per-model attribution arrays into a single array.

    Args:
        attributions (list of numpy.ndarray): List of attribution matrices,
            each of shape ``(n_cells, n_genes)``.
        strategy (str): Aggregation method — ``'mean'`` (default), ``'sum'``,
            or ``'median'``.

    Returns:
        numpy.ndarray: Aggregated attribution matrix of shape
            ``(n_cells, n_genes)``.
    """
    if strategy == 'mean':
        return np.mean(attributions, axis = 0)
    elif strategy == 'sum':
        return np.sum(attributions, axis = 0)
    elif strategy == 'median':
        return np.median(attributions, axis = 0)
    else:
        # Default to mean aggregation
        return np.mean(attributions, axis = 0)


def _get_explainer(model, explainer_type, raw=False):
    """Wrap a model in the requested Captum explainer.

    Args:
        model (torch.nn.Module): Trained autoencoder with a single-tensor
            forward pass (mode flag already set).
        explainer_type (str): One of ``'GradientShap'``, ``'GuidedBackprop'``,
            ``'Deconvolution'``.
        raw (bool): For ``GradientShap`` only — if ``True``, sets
            ``multiply_by_inputs=False``. Defaults to ``False``.

    Returns:
        tuple: ``(explainer, explainer_mode)`` where ``explainer_mode`` is
            ``'shap-like'`` for GradientShap or ``'lrp-like'`` for the others.

    Raises:
        ValueError: If ``explainer_type`` is not recognised.
    """

    if explainer_type in ['GuidedBackprop', 'Deconvolution']:
        explainer_mode = 'lrp-like'
    else:
        explainer_mode = 'shap-like'


    if explainer_type == 'GuidedBackprop': #fast
        explainer = GuidedBackprop(model)
    elif explainer_type == 'GradientShap': #fast
        if raw:
            explainer = GradientShap(model, multiply_by_inputs=False)
        else:
            explainer = GradientShap(model, multiply_by_inputs=True)

    elif explainer_type == 'Deconvolution': #fast
        explainer = Deconvolution(model)
    else:
        raise ValueError('no such method')

    return explainer, explainer_mode



def shuffle_each_column_independently(tensor):
    """
    Shuffles each column of a 2D PyTorch tensor independently.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: A new tensor with each of its columns independently shuffled.
    """
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional to shuffle columns.")

    # Create an empty tensor of the same size to store the shuffled columns
    shuffled_tensor = torch.empty_like(tensor)

    # Iterate through each column, shuffle it, and place it in the new tensor
    for i in range(tensor.size(1)):
        column = tensor[:, i]
        idx = torch.randperm(column.nelement())
        shuffled_tensor[:, i] = column[idx]

    return shuffled_tensor


def attribution_one_target(
        target_gene,
        lrp_model,
        input_data,
        xai_type='lrp-like',
        background_type = 'zeros') -> list:

    """Compute Captum attributions for a single target gene across all models.

    Iterates over every explainer in ``lrp_model`` and calls ``.attribute()``
    with the appropriate signature depending on ``xai_type``. The background
    tensor used for SHAP-style attribution is constructed from ``input_data``
    according to ``background_type``.

    Args:
        target_gene (int): Index of the target gene in the expression matrix.
        lrp_model (list): List of Captum explainer objects (one per model in zoo).
        input_data (torch.Tensor): Input data tensor on CUDA, shape
            ``(n_cells, n_genes)``.
        xai_type (str): ``'lrp-like'`` or ``'shap-like'``.
        background_type (str): Background for SHAP-like methods —
            ``'zeros'`` (default), ``'randomize'`` (shuffle columns), or
            ``'data'`` (use input as its own baseline).

    Returns:
        list of numpy.ndarray: One attribution matrix per model, each of shape
            ``(n_cells, n_genes)``.
    """

    if background_type == 'randomize':
        background = shuffle_each_column_independently(input_data)
    elif background_type == 'zeros':
        background = torch.zeros((1, input_data.shape[1]))
        background = background.cuda()
    elif background_type == 'data':
        background = input_data
    else:
        background = torch.zeros((1, input_data.shape[1]))
        background = background.cuda()


    attributions_list = []
    for m in range(len(lrp_model)):
        # Randomize backgorund for each round
        model = lrp_model[m]
        #for _ in range(num_iterations):
        if xai_type == 'lrp-like':
            attribution = model.attribute(input_data, target=target_gene)

        elif xai_type == 'shap-like':
            attribution = model.attribute(input_data, baselines = background, target = target_gene)

        attributions_list.append(attribution.detach().cpu().numpy())
    return attributions_list



def inferrence(models, data_train_full_tensor, gene_names, xai_method='GradientShap', background_type='zeros', backing_file='grn_adata.h5', return_in_memory=False):
    """Compute the full GRN for all target genes using an autoencoder ensemble.

    For each gene acting as an attribution target, attribution scores are
    computed across all models in the ensemble, aggregated by mean, and either
    written as a Parquet shard or collected in memory. The resulting AnnData
    object has cells as observations and directed edges (SourceGene_TargetGene)
    as variables.

    When ``backing_file`` is not ``None``, one Parquet file per target gene is
    written to ``<dirname(backing_file)>/grn/<gene>.parquet`` and ``grn_adata.X``
    is a zero-shape placeholder. Use ``retrieve_top_edges()`` from
    ``netmap.utils.data_utils`` to load attribution values on demand.

    Args:
        models (list): List of trained autoencoder model instances.
        data_train_full_tensor (torch.Tensor): Input expression data on CUDA.
        gene_names (numpy.ndarray): Gene name array; order must match tensor columns.
        xai_method (str): XAI method — ``'GradientShap'`` (default),
            ``'GuidedBackprop'``, or ``'Deconvolution'``.
        background_type (str): Background for SHAP-like attributions.
            Defaults to ``'zeros'``.
        backing_file (str or None): Path prefix for output. Parquet shards are
            written to ``<dirname(backing_file)>/grn/<gene>.parquet``.
            Set to ``None`` to keep the full matrix in memory (high RAM usage).
        return_in_memory (bool): If ``backing_file`` is set and this is ``True``,
            load the HDF5 file back into memory before returning. Defaults to
            ``False``.

    Returns:
        anndata.AnnData: GRN object with cells as obs and edges as var; ``X`` is
            an empty placeholder when backed by Parquet.
    """

    tms = []
    name_list = []
    target_names = []


    for trained_model in models:
        trained_model.forward_mu_only = True
        explainer, xai_type = _get_explainer(trained_model, xai_method, raw=False)
        tms.append(explainer)

    attributions = []

    rows = data_train_full_tensor.shape[0]
    cols = data_train_full_tensor.shape[1]
    cols_grn = cols*cols

    collect_sums = []
    collect_means = []



    if backing_file is not None:

        # Configuration
        output_dir = op.dirname(backing_file)
        output_dir = op.join(output_dir, 'grn')
        os.makedirs(output_dir, exist_ok=True)

        name_list =  []
        name = 'attr'

        for i in range(cols):
            ## Create name vector
            name_list = name_list + list(gene_names)
            target_names = target_names+[gene_names[i]] *len(gene_names)

        column_names = [f'{s}_{t}' for s,t in zip(name_list, target_names)]

        schema = pa.schema([(name, pa.float32()) for name in column_names])

        # Loop through your column-wise groups
        for g in tqdm(range(data_train_full_tensor.shape[1])):
            # Generate your column-wise chunk (shape: [rows, cols])
            attributions_list = attribution_one_target(
                g, tms, data_train_full_tensor,
                xai_type=xai_type, background_type=background_type
            )

            attributions_list = aggregate_attributions(attributions_list, strategy='mean')

            collect_sums.append(np.sum(attributions_list, axis=0))
            collect_means.append(np.mean(attributions_list, axis=0))

            # 2. Convert the column-chunk to a PyArrow Table
            # Map the numpy chunk to the specific column names for this group 'g'
            current_col_names = column_names[g*cols : (g+1)*cols]

            # We create a table where each slice of the numpy array is a column
            chunk_table = pa.table(
                [attributions_list[:, i] for i in range(attributions_list.shape[1])],
                names=current_col_names
            )

            # 3. Write this specific column-group to a Parquet file
            # In a dataset, these will be "sharded" columns
            file_path = os.path.join(output_dir, f"{gene_names[g]}.parquet")
            pq.write_table(chunk_table, file_path)



    else:
        for g in tqdm(range(data_train_full_tensor.shape[1])):
                attributions_list = attribution_one_target(
                    g,
                    tms,
                    data_train_full_tensor,
                    xai_type=xai_type,
                    background_type= background_type)



                attributions_list = aggregate_attributions(attributions_list, strategy='mean')
                collect_sums.append(np.sum(attributions_list, axis = 0))
                collect_means.append(np.mean(attributions_list, axis = 0))

                attributions.append(attributions_list)

        attributions = np.hstack(attributions)

    name_list = []
    target_names = []
    for i in range(cols):
        ## Create name vector
        name_list = name_list + list(gene_names)
        target_names = target_names+[gene_names[i]] *len(gene_names)


    index_list = [f"{s}_{t}" for (s, t) in zip(name_list, target_names)]
    cou = pd.DataFrame({'index': index_list, 'source':name_list, 'target':target_names})
    cou = cou.set_index('index')
    cou['edge_sums'] = np.concatenate(collect_sums)
    cou['edge_means'] = np.concatenate(collect_means)

    if backing_file is not None:
        if return_in_memory:
            with h5py.File(backing_file, 'r+') as f:
                dset = f['data']
                grn_adata  = ad.AnnData(dset, uns = {'backing_file': backing_file}, var = cou)
                grn_adata = grn_adata.to_memory()

        else:
            grn_adata  = ad.AnnData(shape = (rows, cols_grn), uns = {'backing_file': backing_file}, var = cou)
    else:
        grn_adata = attribution_to_anndata(attributions, var=cou)

    return grn_adata


def inferrence_h5py(models, data_train_full_tensor, gene_names, xai_method='GradientShap', background_type='zeros', backing_file='grn_adata.h5', return_in_memory=False):
    """Compute the full GRN and back it with an HDF5 file instead of Parquet.

    Functionally equivalent to ``inferrence()``, but attribution data is
    written into a single HDF5 dataset (key ``'data'``, shape
    ``(cells, genes^2)``, chunked by ``(cells, genes)``) rather than
    per-gene Parquet shards.

    Args:
        models (list): List of trained PyTorch autoencoder models. The
            ``forward_mu_only`` flag is set to ``True`` on each model
            internally before wrapping it in a Captum explainer.
        data_train_full_tensor (torch.Tensor): Expression data tensor of shape
            ``(cells, genes)`` on CUDA.
        gene_names (array-like): Ordered gene names corresponding to columns
            of ``data_train_full_tensor``.
        xai_method (str): Captum XAI method to use. One of
            ``'GradientShap'`` (default), ``'GuidedBackprop'``,
            ``'Deconvolution'``.
        background_type (str): Baseline strategy for SHAP-style attribution.
            One of ``'zeros'`` (default), ``'randomize'``, ``'data'``.
        backing_file (str or None): Path to the HDF5 output file. If not
            ``None``, the file is created (overwriting any existing file) and
            attribution data is streamed into it column-block by column-block.
            The path is stored in ``grn_adata.uns['backing_file']`` so that
            ``return_grn_adata_to_memory()`` can reload it. If ``None``, all
            attributions are accumulated in memory.
        return_in_memory (bool): Only relevant when ``backing_file`` is not
            ``None``. If ``True``, the HDF5 dataset is read back into memory
            and ``grn_adata.X`` is fully populated before returning.
            Default ``False``.

    Returns:
        anndata.AnnData: GRN object with shape ``(cells, genes^2)``.
            ``obs`` corresponds to cells; ``var`` contains directed edge
            metadata with columns ``source``, ``target``, ``edge_sums``, and
            ``edge_means``. When backed by HDF5 and ``return_in_memory=False``,
            ``X`` is a zero-shape placeholder.
    """

    tms = []
    name_list = []
    target_names = []


    for trained_model in models:
        trained_model.forward_mu_only = True
        explainer, xai_type = _get_explainer(trained_model, xai_method, raw=False)
        tms.append(explainer)

    attributions = []

    rows = data_train_full_tensor.shape[0]
    cols = data_train_full_tensor.shape[1]
    cols_grn = cols*cols

    collect_sums = []
    collect_means = []

    if backing_file is not None:
        with h5py.File(backing_file, 'w') as f:

            dset = f.create_dataset(
                'data',
                shape=(rows, cols_grn),
                dtype='float32',
                chunks=(rows, cols)
            )

            for g in tqdm(range(data_train_full_tensor.shape[1])):
                attributions_list = attribution_one_target(
                    g,
                    tms,
                    data_train_full_tensor,
                    xai_type=xai_type,
                    background_type= background_type)



                attributions_list = aggregate_attributions(attributions_list, strategy='mean')
                collect_sums.append(np.sum(attributions_list, axis = 0))
                collect_means.append(np.mean(attributions_list, axis = 0))
                dset[:, (g*cols): ((g+1)*cols)] = attributions_list

    else:
        for g in tqdm(range(data_train_full_tensor.shape[1])):
                attributions_list = attribution_one_target(
                    g,
                    tms,
                    data_train_full_tensor,
                    xai_type=xai_type,
                    background_type= background_type)



                attributions_list = aggregate_attributions(attributions_list, strategy='mean')
                collect_sums.append(np.sum(attributions_list, axis = 0))
                collect_means.append(np.mean(attributions_list, axis = 0))

                attributions.append(attributions_list)

        attributions = np.hstack(attributions)

    for i in range(cols):
        ## Create name vector
        name_list = name_list + list(gene_names)
        target_names = target_names+[gene_names[i]] *len(gene_names)


    index_list = [f"{s}_{t}" for (s, t) in zip(name_list, target_names)]
    cou = pd.DataFrame({'index': index_list, 'source':name_list, 'target':target_names})
    cou = cou.set_index('index')
    cou['edge_sums'] = np.concatenate(collect_sums)
    cou['edge_means'] = np.concatenate(collect_means)

    if backing_file is not None:
        if return_in_memory:
            with h5py.File(backing_file, 'r+') as f:
                dset = f['data']
                grn_adata  = ad.AnnData(dset, uns = {'backing_file': backing_file}, var = cou)
                grn_adata = grn_adata.to_memory()

        else:
            grn_adata  = ad.AnnData(shape = (rows, cols_grn), uns = {'backing_file': backing_file}, var = cou)
    else:
        grn_adata = attribution_to_anndata(attributions, var=cou)

    return grn_adata


def return_grn_adata_to_memory(grn_adata):
    """Load the full attribution matrix from the HDF5 backing file into memory.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData with ``uns['backing_file']``
            pointing to an HDF5 file written by :func:`inferrence_h5py`.

    Returns:
        anndata.AnnData: The same object with ``X`` populated in memory.
    """
    with h5py.File(grn_adata.uns['backing_file'], 'r+') as f:
        dset = f['data']
        grn_adata.X = dset
        grn_adata = grn_adata.to_memory()
    return grn_adata


def attribution_one_model(
        lrp_model,
        input_data,
        xai_type='lrp-like',
        background_type = 'randomize'):


    """Compute attributions for all target genes using a single model.

    Iterates over every gene index as an attribution target and calls
    ``lrp_model.attribute()`` with the appropriate signature, then
    horizontally stacks the per-target arrays into one complete attribution
    matrix for the model.

    Args:
        lrp_model: A single Captum explainer instance (e.g. GradientShap or
            GuidedBackprop) wrapping one trained autoencoder.
        input_data (torch.Tensor): Expression tensor of shape
            ``(cells, genes)`` on CUDA.
        xai_type (str): Explainer mode. One of ``'lrp-like'`` (no baseline
            needed) or ``'shap-like'`` (baseline required). Default
            ``'lrp-like'``.
        background_type (str): Strategy for constructing the baseline tensor
            used by SHAP-style methods. One of:

            - ``'randomize'``: ``input_data`` with each column independently
              shuffled (default).
            - ``'zeros'``: a single all-zero row on CUDA.
            - ``'data'``: use ``input_data`` itself as the baseline.

            Any unrecognised value falls back to ``'zeros'``.

    Returns:
        np.ndarray: Attribution matrix of shape ``(cells, genes^2)`` where
        columns are ordered by target gene (outer) then source gene (inner),
        matching the edge ordering used in ``grn_adata.var``.
    """

    attributions_list = []

    # Randomize backgorund for each round
    if background_type == 'randomize':
        background = shuffle_each_column_independently(input_data)
    elif background_type == 'zeros':
        background = torch.zeros((1, input_data.shape[1]))
        background = background.cuda()
    elif background_type == 'data':
        background = input_data
    else:
        background = torch.zeros((1, input_data.shape[1]))
        background = background.cuda()


    for target_gene in tqdm(range(input_data.shape[1])):

        #for _ in range(num_iterations):
        if xai_type == 'lrp-like':
            attribution = lrp_model.attribute(input_data, target=target_gene)


        elif xai_type == 'shap-like':
            attribution = lrp_model.attribute(input_data, baselines = background, target = target_gene)

        attributions_list.append(attribution.detach().cpu().numpy())

    attributions = np.hstack(attributions_list)
    return attributions


def inferrence_model_wise(models, data_train_full_tensor, gene_names, xai_method='GradientShap', background_type='zeros'):
    """Compute the full GRN by processing one complete model at a time.

    Unlike ``inferrence()``, which iterates over target genes across all
    models, this function processes each model in full (all target genes) and
    accumulates a running sum of attribution matrices. The final matrix is
    normalised implicitly by the last model added. This approach may reduce
    peak memory when the number of genes is small relative to the number of
    models.

    Args:
        models (list): List of trained PyTorch autoencoder models. The
            ``forward_mu_only`` flag is set to ``True`` on each model
            internally.
        data_train_full_tensor (torch.Tensor): Expression tensor of shape
            ``(cells, genes)`` on CUDA.
        gene_names (array-like): Ordered gene names corresponding to columns
            of ``data_train_full_tensor``.
        xai_method (str): Captum XAI method to use. One of
            ``'GradientShap'`` (default), ``'GuidedBackprop'``,
            ``'Deconvolution'``.
        background_type (str): Baseline strategy for SHAP-style attribution.
            One of ``'zeros'`` (default), ``'randomize'``, ``'data'``.

    Returns:
        anndata.AnnData: In-memory GRN object with shape
        ``(cells, genes^2)``. ``var`` contains directed edge metadata with
        columns ``source`` and ``target``.
    """

    tms = []

    cou  = [[f'{tup[0]}_{tup[1]}', tup[0], tup[1]] for tup in itertools.product(gene_names, gene_names)]
    cou = pd.DataFrame(cou)
    cou.columns = ['index', 'source', 'target']
    cou = cou.set_index('index')


    for trained_model in models:
        trained_model.forward_mu_only = True
        explainer, xai_type = _get_explainer(trained_model, xai_method)
        tms.append(explainer)

    attributions = {}
    attribution_collector = None
    keynames = []


    for m in range(len(tms)):

        # get one complete attribution
        current_attribution = attribution_one_model(
            tms[m],
            data_train_full_tensor,
            xai_type=xai_type,
            background_type = background_type)


        if attribution_collector is not None:
            # add current attribution to the collector
            attribution_collector =  aggregate_attributions([attribution_collector, current_attribution], strategy='sum')


        else:
            attribution_collector = current_attribution


    grn_adata = attribution_to_anndata(attribution_collector, var=cou)

    return grn_adata




def inferrence_model_wise_level(models, data_train_full_tensor, gene_names, xai_method='GradientShap', n_models=[10, 25, 50], background_type='zeros'):
    """Compute the GRN model-wise and save intermediate aggregates as layers.

    Processes one complete model at a time (same order as
    ``inferrence_model_wise()``), but additionally saves a snapshot of the
    mean attribution matrix at each model-count checkpoint listed in
    ``n_models``. Snapshots after the first checkpoint are stored as named
    layers in the returned AnnData object so that downstream analysis can
    compare GRN stability across ensemble sizes.

    Args:
        models (list): List of trained PyTorch autoencoder models. The
            ``forward_mu_only`` flag is set to ``True`` on each model
            internally.
        data_train_full_tensor (torch.Tensor): Expression tensor of shape
            ``(cells, genes)`` on CUDA.
        gene_names (array-like): Ordered gene names corresponding to columns
            of ``data_train_full_tensor``.
        xai_method (str): Captum XAI method to use. One of
            ``'GradientShap'`` (default), ``'GuidedBackprop'``,
            ``'Deconvolution'``.
        n_models (list of int): Checkpoints (1-based model counts) at which
            the current running mean attribution matrix is saved as a layer.
            For example, ``[10, 25, 50]`` saves snapshots after models 10,
            25, and 50 have been processed. The first checkpoint becomes
            ``grn_adata.X``; subsequent checkpoints are stored as
            ``grn_adata.layers['aggregated_<n>']``. Default ``[10, 25, 50]``.
        background_type (str): Baseline strategy for SHAP-style attribution.
            One of ``'zeros'`` (default), ``'randomize'``, ``'data'``.

    Returns:
        anndata.AnnData or None: GRN object with shape ``(cells, genes^2)``
        where ``X`` holds the mean attributions at the first checkpoint and
        each additional checkpoint is stored as a named layer. Returns
        ``None`` if no checkpoint in ``n_models`` was reached (e.g. fewer
        models were provided than the smallest value in ``n_models``).
    """

    tms = []

    cou  = [[f'{tup[0]}_{tup[1]}', tup[0], tup[1]] for tup in itertools.product(gene_names, gene_names)]
    cou = pd.DataFrame(cou)
    cou.columns = ['index', 'source', 'target']
    cou = cou.set_index('index')


    for trained_model in models:
        trained_model.forward_mu_only = True
        explainer, xai_type = _get_explainer(trained_model, xai_method)
        tms.append(explainer)

    attributions = {}
    attribution_collector = None
    keynames = []


    for m in range(len(tms)):

        # get one complete attribution
        current_attribution = attribution_one_model(
            tms[m],
            data_train_full_tensor,
            xai_type=xai_type,
            background_type = background_type)


        if attribution_collector is not None:
            # add current attribution to the collector
            attribution_collector =  aggregate_attributions([attribution_collector, current_attribution], strategy='sum')


        else:
            attribution_collector = current_attribution


        try:
            if (m+1) in n_models:
                # dont reset, just save the correct matrix
                attributions[f'aggregated_{(m+1)}'] = attribution_collector/(m+1)
                keynames.append(f'aggregated_{(m+1)}')
        except:
            pass


    # top_egde_collector = pd.DataFrame(top_egde_collector)
    # top_egde_collector['variance_area'] = top_egde_collector.iloc[:, 0:10].var(axis=1)
    # top_egde_collector['mean_area'] = top_egde_collector.iloc[:, 0:10].mean(axis=1)
    # top_egde_collector['zscore'] = scipy.stats.zscore(top_egde_collector['mean_area'])

    # cou = cou.merge(top_egde_collector, left_index = True, right_on='edge_key')


    if len(keynames)>0:
        grn_adata = attribution_to_anndata(attributions[keynames[0]], var=cou)

        for k in keynames[1:len(keynames)]:
            # add remaining versions as masks
            grn_adata.layers[k] = attributions[k]
        return grn_adata
    else:
        return None
