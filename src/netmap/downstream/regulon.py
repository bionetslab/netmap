"""Regulon selection and signed regulon construction from GRN attribution data.

:func:`select_top_edges` and :func:`select_top_edges_signed` select high-confidence
edges per source TF per cluster, filtered by neighbourhood co-expression support.
The signed variant uses cluster-wise Spearman correlation to split edges into
positive and negative regulons.
"""

import logging
import warnings
from functools import reduce
from itertools import chain, product, combinations
from typing import List, Optional, Tuple, Union, Dict
import json
import os
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, ranksums
#from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import networkx as nx
import requests
from pyvis.network import Network
import pyucell as ucell
import scipy.sparse as scs


from netmap.downstream.clustering import process, spectral_clustering, downstream_recipe
from netmap.downstream.edge_selection import add_top_edge_annotation_global


from itertools import combinations
from collections import Counter


def select_top_edges(gene_inter_adata, adata, top_per_source=10, col_cluster='leiden_remap', min_reg_size=10, verbose=True, return_copy=False, tf_column=None, min_edge_support=0.5):
    """Select the top-attribution edges per source TF per cluster.

    For each cluster, computes per-cell mean attribution across cluster cells for each
    source gene's edges, filters by ``min_edge_support`` from the mask layer, and keeps
    the top ``top_per_source`` edges per source. Returns the processed edge dict.

    Args:
        gene_inter_adata (anndata.AnnData): GRN AnnData with ``X`` (attributions),
            ``layers['mask']``, and cluster labels in ``obs``.
        adata (anndata.AnnData): Expression AnnData (currently unused, reserved).
        top_per_source (int): Max edges per source TF per cluster. Defaults to 10.
        col_cluster (str): Cluster column in ``obs``. Defaults to ``'leiden_remap'``.
        min_reg_size (int): Minimum edges required to keep a regulon. Defaults to 10.
        verbose (bool): Print progress. Defaults to ``True``.
        return_copy (bool): Unused, reserved. Defaults to ``False``.
        tf_column (str or None): var column flagging TF rows; restricts sources to TFs.
        min_edge_support (float): Minimum mask fraction required per edge. Defaults to 0.5.

    Returns:
        dict: Nested result from :func:`process_cell_edges`.
    """
    clusters = list(np.unique(gene_inter_adata.obs[col_cluster]))
    keep_edges_dict = {}

    for c in clusters:
        Keep_edges = [] # Now will store tuples: (edge_name, sum_val)
        if verbose: print(f"Selecting targets for cluster: {c}")

        cells_c = gene_inter_adata.obs[col_cluster] == c
        gene_inter_adata.var['edge_support_c'] = (
            gene_inter_adata.layers['mask'][cells_c, :].mean(axis=0)
        )

        if tf_column is not None:
            tfs = gene_inter_adata.var[gene_inter_adata.var[tf_column]]['source'].unique()
            source_list = set(gene_inter_adata.var["source"].unique()).intersection(set(tfs))
        else:
            source_list = gene_inter_adata.var["source"].unique()

        for source in source_list:
            df_targets = gene_inter_adata.var[
                    (gene_inter_adata.var['source'] == source) &
                    (gene_inter_adata.var['edge_support_c'] >= min_edge_support)].copy()

            # Calculate sum and sort
            df_targets['sum_of_edge'] = gene_inter_adata[cells_c, df_targets.index].X.mean(axis=0)
            df_targets = df_targets.sort_values('sum_of_edge', ascending=False).head(top_per_source)

            if len(df_targets) >= min_reg_size:
                for _, row in df_targets.iterrows():
                    edge_str = f"{source}_{row['target']}"
                    Keep_edges.append((edge_str, row['sum_of_edge']))

            keep_edges_dict[c] = Keep_edges

    return process_cell_edges(keep_edges_dict)


def process_cell_edges(keep_edges_with_vals):
    """Organise per-cluster edge lists into a nested results dictionary.

    Builds two views of the data: ``'unique'`` (edges not present in any other cluster)
    and ``'all'`` (all selected edges). Each cluster entry contains an ``'edges'``
    DataFrame with ``source``, ``target``, ``sum_of_edge`` columns and a ``'summary'``
    DataFrame with source gene counts.

    Args:
        keep_edges_with_vals (dict): Mapping ``{cluster: [(edge_str, mean_val), ...]}``

    Returns:
        dict: ``{'unique': {cluster: {'edges': DataFrame, 'summary': DataFrame}}, 'all': ...}``.
    """
    results = {'unique': {}, 'all': {}}
    all_cells = list(keep_edges_with_vals.keys())

    def get_source_summary(edge_list):
        # edge_list is list of (name, val)
        sources = [e[0].split('_')[0] for e in edge_list]
        source_dict = dict(Counter(sources))
        sources_df = pd.DataFrame({'source': list(source_dict.keys()), 'count': list(source_dict.values())}).sort_values('count', ascending=False)
        return sources_df

    for cell in all_cells:
        # Convert to dict for easy lookup by edge name string
        current_edges_dict = {name: val for name, val in keep_edges_with_vals[cell]}

        # Calculate Uniques based on the edge name string
        others_names = set()
        for c in all_cells:
            if c != cell:
                others_names.update([e[0] for e in keep_edges_with_vals[c]])

        unique_names = set(current_edges_dict.keys()) - others_names

        # Helper to build DF with sum column
        def build_df(name_set, lookup_dict):
            data = []
            for name in name_set:
                source, target = name.split('_', 1)
                data.append([source, target, lookup_dict[name]])
            return pd.DataFrame(data, columns=['source', 'target', 'sum_of_edge'])

        results['unique'][cell] = {
            'edges': build_df(unique_names, current_edges_dict),
            'summary': get_source_summary([(n, current_edges_dict[n]) for n in unique_names])
        }

        results['all'][cell] = {
            'edges': build_df(current_edges_dict.keys(), current_edges_dict),
            'summary': get_source_summary(keep_edges_with_vals[cell])
        }

    return results


def process_cell_edges_signed(keep_edges_pos, keep_edges_neg):
    """Build the positive/negative regulon result structure from per-cluster edge lists.

    Args:
        keep_edges_pos (dict): Mapping of cluster label to a list of
            ``(edge_str, attribution_value)`` tuples for positively correlated edges.
        keep_edges_neg (dict): Mapping of cluster label to a list of
            ``(edge_str, attribution_value)`` tuples for negatively correlated edges.

    Returns:
        dict: Nested result with keys ``'unique'`` and ``'all'``, each mapping cluster
        label to a dict with keys ``'positive'`` and ``'negative'``, each of which
        contains:

            - ``'edges'`` (pd.DataFrame): Columns ``source``, ``target``,
              ``sum_of_edge``.
            - ``'summary'`` (pd.DataFrame): Columns ``source``, ``count``.
    """
    results = {'unique': {}, 'all': {}}
    all_cells = list(keep_edges_pos.keys())

    def get_source_summary(edge_list):
        sources = [e[0].split('_')[0] for e in edge_list]
        source_dict = dict(Counter(sources))
        return pd.DataFrame(
            {'source': list(source_dict.keys()), 'count': list(source_dict.values())}
        ).sort_values('count', ascending=False)

    def build_df(name_set, lookup_dict):
        data = []
        for name in name_set:
            source, target = name.split('_', 1)
            data.append([source, target, lookup_dict[name]])
        return pd.DataFrame(data, columns=['source', 'target', 'sum_of_edge'])

    for cell in all_cells:
        pos_dict = dict(keep_edges_pos[cell])
        neg_dict = dict(keep_edges_neg[cell])

        others_pos = set()
        others_neg = set()
        for c in all_cells:
            if c != cell:
                others_pos.update([e[0] for e in keep_edges_pos[c]])
                others_neg.update([e[0] for e in keep_edges_neg[c]])

        unique_pos = set(pos_dict.keys()) - others_pos
        unique_neg = set(neg_dict.keys()) - others_neg

        results['unique'][cell] = {
            'positive': {
                'edges': build_df(unique_pos, pos_dict),
                'summary': get_source_summary([(n, pos_dict[n]) for n in unique_pos]),
            },
            'negative': {
                'edges': build_df(unique_neg, neg_dict),
                'summary': get_source_summary([(n, neg_dict[n]) for n in unique_neg]),
            },
        }
        results['all'][cell] = {
            'positive': {
                'edges': build_df(pos_dict.keys(), pos_dict),
                'summary': get_source_summary(keep_edges_pos[cell]),
            },
            'negative': {
                'edges': build_df(neg_dict.keys(), neg_dict),
                'summary': get_source_summary(keep_edges_neg[cell]),
            },
        }

    return results


def _collect_signed_edges(df_targets, source, spearman_col, min_reg_size, neutral_threshold):
    """Split a source gene's target edges into positive and negative lists by Spearman sign.

    Args:
        df_targets (pd.DataFrame): Edge candidates for one source with a ``spearman_col``
            column and ``sum_of_edge``.
        source (str): Source TF gene name.
        spearman_col (str): Column name in df_targets containing Spearman values.
        min_reg_size (int): Minimum edges per sign group; empty list returned if not met.
        neutral_threshold (float): Edges with |spearman| < this are excluded.

    Returns:
        tuple: ``(pos_edges, neg_edges)`` each a list of ``(edge_str, mean_val)`` tuples.
    """
    df_pos = df_targets[df_targets[spearman_col] >= neutral_threshold]
    df_neg = df_targets[df_targets[spearman_col] <= -neutral_threshold]
    pos = (
        [(f"{source}_{r['target']}", r['sum_of_edge']) for _, r in df_pos.iterrows()]
        if len(df_pos) >= min_reg_size else []
    )
    neg = (
        [(f"{source}_{r['target']}", r['sum_of_edge']) for _, r in df_neg.iterrows()]
        if len(df_neg) >= min_reg_size else []
    )
    return pos, neg


def select_top_edges_signed(gene_inter_adata, top_per_source=10, col_cluster='leiden_remap',
                             min_reg_size=10, verbose=True, tf_column=None,
                             min_edge_support=0.5, neutral_threshold=0.05):
    """Select top edges per source TF per cluster and split them into positive and
    negative regulons based on cluster-wise Spearman correlation.

    Requires ``add_cluster_wise_spearman`` to have been called first so that
    ``{cluster}_spearman`` columns are present in ``gene_inter_adata.var``.

    Edges with |spearman| < neutral_threshold are excluded (neutral effect).

    Args:
        gene_inter_adata (anndata.AnnData): GRN AnnData with mask layer and
            '{cluster}_spearman' columns in .var.
        top_per_source (int): Maximum edges to consider per source TF per cluster.
        col_cluster (str): Cluster column in obs.
        min_reg_size (int): Minimum edges a sign-group must have to be kept.
        verbose (bool): Print progress.
        tf_column (str or None): var column flagging TF rows; restricts sources to TFs.
        min_edge_support (float): Minimum mask support fraction required per edge.
        neutral_threshold (float): Edges with |spearman| < this value are excluded.

    Returns:
        dict: {'unique': {cluster: {'positive': ..., 'negative': ...}},
               'all':    {cluster: {'positive': ..., 'negative': ...}}}
    """
    clusters = list(np.unique(gene_inter_adata.obs[col_cluster]))
    keep_edges_pos = {}
    keep_edges_neg = {}

    for c in clusters:
        if verbose:
            print(f"Selecting targets for cluster: {c}")

        cells_c = gene_inter_adata.obs[col_cluster] == c
        gene_inter_adata.var['edge_support_c'] = (
            gene_inter_adata.layers['mask'][cells_c, :].mean(axis=0)
        )
        spearman_col = f'{c}_spearman'

        if tf_column is not None:
            tfs = gene_inter_adata.var[gene_inter_adata.var[tf_column]]['source'].unique()
            source_list = set(gene_inter_adata.var["source"].unique()).intersection(set(tfs))
        else:
            source_list = gene_inter_adata.var["source"].unique()

        pos_edges, neg_edges = [], []
        for source in source_list:
            df_targets = gene_inter_adata.var[
                (gene_inter_adata.var['source'] == source) &
                (gene_inter_adata.var['edge_support_c'] >= min_edge_support)
            ].copy()
            df_targets['sum_of_edge'] = gene_inter_adata[cells_c, df_targets.index].X.mean(axis=0)
            df_targets = df_targets.sort_values('sum_of_edge', ascending=False).head(top_per_source)

            pos, neg = _collect_signed_edges(df_targets, source, spearman_col, min_reg_size, neutral_threshold)
            pos_edges.extend(pos)
            neg_edges.extend(neg)

        keep_edges_pos[c] = pos_edges
        keep_edges_neg[c] = neg_edges

    return process_cell_edges_signed(keep_edges_pos, keep_edges_neg)


def compute_signatures_UCell_scores(selected_edges, adata, key='unique') -> pd.DataFrame:
    """Compute UCell gene set enrichment scores for regulon signatures.

    Args:
        selected_edges (dict): Nested edge dict from :func:`select_top_edges`.
        adata (anndata.AnnData): Expression AnnData; UCell scores are added to
            ``adata.obs``.
        key (str): Top-level key of ``selected_edges`` to use — ``'unique'`` or
            ``'all'``. Defaults to ``'unique'``.

    Returns:
        pd.DataFrame: UCell scores per cell per regulon, columns named
        ``{cluster}_{TF}``.
    """

    all_signatures = {}
    for ct in selected_edges[key]:
        sign = selected_edges[key][ct]['edges'].groupby('source')['target'].apply(list).to_dict()
        sign  = {f"{ct}_{k}": v for k, v in sign.items()}
        all_signatures = all_signatures | sign

    ucell.compute_ucell_scores(adata, signatures=all_signatures, n_jobs=1)
    data_ucell = adata.obs.filter(like='_UCell')
    data_ucell.columns = [x.replace('_UCell', '') for x in data_ucell.columns]

    return data_ucell


def aggregate_edges(selected_edges, grn_adata, key='unique', grouping='source') -> pd.DataFrame:
    """Aggregate regulon attribution scores across edges.

    For each cluster and source gene, averages attribution values over all selected
    edges to produce a single regulon activity score per cell.

    Args:
        selected_edges (dict): Nested edge dict from :func:`select_top_edges`.
        grn_adata (anndata.AnnData): GRN AnnData with attribution values in ``X``.
        key (str): Top-level key to use — ``'unique'`` or ``'all'``. Defaults to
            ``'unique'``.
        grouping (str): Column to group edges by — ``'source'`` or ``'target'``.
            Defaults to ``'source'``.

    Returns:
        pd.DataFrame: Regulon activity matrix, shape ``(n_cells, n_regulons)``.
    """

    regulons = {}
    for ct in selected_edges[key]:
        print(ct)
        sign = selected_edges[key][ct]['edges'].groupby(grouping).apply(lambda x: (x['source'] + "_" + x['target']).tolist()).to_dict()
        for g in sign:
            regulons[f'{ct}_{g}'] = grn_adata[:, sign[g]].X.mean(axis=1)
    regulons = pd.DataFrame(regulons)
    return regulons


def aggregate_edges_arbitrary(grn_adata, edge_dict) -> pd.DataFrame:
    """Aggregate attribution scores for arbitrary named edge sets.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData with attribution values in ``X``.
        edge_dict (dict): Mapping ``{regulon_name: [edge_list]}``.

    Returns:
        pd.DataFrame: Regulon activity matrix, shape ``(n_cells, n_regulons)``.
    """

    regulons = {}
    for name in edge_dict:
        if scs.issparse(grn_adata.X):
            regulons[f'{name}'] = np.asarray(grn_adata[:, list(edge_dict[name])].X.mean(axis=1))[:,0]
        else:
            regulons[f'{name}'] = grn_adata[:, list(edge_dict[name])].X.mean(axis=1)


    regulons = pd.DataFrame(regulons)
    return regulons


import pandas as pd
from itertools import combinations

def get_overlapping_signatures(all_regulons):
    """Compute pairwise overlap status of target genes across clusters.

    For every pair of clusters, determines for each target gene in their union
    whether it appears in both clusters, only in the first, or only in the second.

    Args:
        all_regulons (pd.DataFrame): Flat regulon DataFrame with at least columns
            ``'cluster'`` and ``'target'``, as produced by
            :func:`make_cluster_regulon_dataframe`.

    Returns:
        pd.DataFrame: Long-form DataFrame with columns ``'celltype_1'``,
        ``'celltype_2'``, ``'status'`` (``'both'``, ``'only {ct1}'``, or
        ``'only {ct2}'``), and ``'gene'``.
    """
    genes_by_ct = all_regulons.groupby('cluster')['target'].apply(set).to_dict()
    all_cts = sorted(genes_by_ct.keys())
    pairwise_results = []

    # 2. Iterate through all pairwise combinations
    for ct1, ct2 in combinations(all_cts, 2):
        genes1 = genes_by_ct[ct1]
        genes2 = genes_by_ct[ct2]

        # Union of all genes in either cell type for this specific pair
        all_genes_in_pair = genes1.union(genes2)

        for gene in all_genes_in_pair:
            # Logic to determine status
            if gene in genes1 and gene in genes2:
                status = 'both'
            elif gene in genes1:
                status = f'only {ct1}'
            else:
                status = f'only {ct2}'

            pairwise_results.append({
                'celltype_1': ct1,
                'celltype_2': ct2,
                'status': status,
                'gene': gene
            })

    # 3. Create the final DataFrame
    pairwise_df = pd.DataFrame(pairwise_results)

    return pairwise_df


def make_cluster_regulon_dataframe(keep_edges):
    """Concatenate all regulon edge DataFrames into a flat DataFrame.

    Iterates over the nested ``keep_edges`` structure (set_type -> cluster ->
    ``{'edges': DataFrame}``) and concatenates all non-empty edge DataFrames into
    a single flat table, adding ``'cluster'`` and ``'set_type'`` columns to
    preserve provenance.

    Args:
        keep_edges (dict): Nested regulon dict as returned by :func:`select_top_edges`
            or :func:`select_top_edges_signed`, with top-level keys such as
            ``'unique'`` and ``'all'``, each mapping cluster labels to a dict
            containing an ``'edges'`` DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame with all columns from the individual
        edge DataFrames plus ``'cluster'`` and ``'set_type'`` columns.
    """
    all_regulons = []
    for un in keep_edges:
        for clu in keep_edges[un]:
            df = keep_edges[un][clu]['edges']
            if df.shape[0] > 0:
                df['cluster'] = clu
                df['set_type'] = un
                all_regulons.append(df)
    all_regulons = pd.concat(all_regulons)
    return all_regulons


def load_edge_dict_from_dataframe(all_regulons):
    """Reconstruct a nested edge dictionary from a flat regulon DataFrame.

    Reverses the transformation performed by :func:`make_cluster_regulon_dataframe`,
    grouping rows back into the ``{set_type: {cluster: {'edges': DataFrame}}}``
    structure. The ``'set_type'`` and ``'cluster'`` columns are dropped from the
    reconstructed edge DataFrames.

    Args:
        all_regulons (pd.DataFrame): Flat DataFrame with at least columns
            ``'set_type'`` and ``'cluster'`` (as produced by
            :func:`make_cluster_regulon_dataframe`) plus whatever edge columns were
            present in the original nested structure.

    Returns:
        dict: Nested dict of the form
        ``{set_type: {cluster: {'edges': pd.DataFrame}}}``.
    """
    keep_edges = {}

    # Group by the metadata columns we added in the forward function
    for (set_type, cluster), group_df in all_regulons.groupby(['set_type', 'cluster']):

        # Initialize the outer dictionary (set_type) if it doesn't exist
        if set_type not in keep_edges:
            keep_edges[set_type] = {}

        # Clean the DataFrame: remove the metadata columns we added
        # and reset the index if necessary to match the original state
        clean_df = group_df.drop(columns=['set_type', 'cluster']).copy()

        # Reconstruct the internal 'edges' dictionary
        keep_edges[set_type][cluster] = {'edges': clean_df}

    return keep_edges


def jaccard_similarity(set1, set2):
    """Compute the Jaccard similarity coefficient between two sets.

    Args:
        set1 (set): First set.
        set2 (set): Second set.

    Returns:
        float: Jaccard similarity in [0, 1]; returns 0 when both sets are empty.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def get_sourcewise_jaccard_regulons(all_signatures, keep_edges, n_top=50):
    """Compute per-source-gene Jaccard similarity matrices across clusters.

    For each cluster, retrieves the top ``n_top`` genes by Wilcoxon rank-sum
    score from ``all_signatures``, then for each top source gene compares its
    target gene lists across all clusters using pairwise Jaccard similarity.
    Returns a square similarity matrix per source gene.

    Note:
        This function references ``grn_adata3`` and ``sc`` from the enclosing or
        module-level scope. This is a known code issue and is not changed here.

    Args:
        all_signatures (anndata.AnnData): AnnData object storing Wilcoxon
            differential regulon results under the ``'wilcoxon'`` key in
            ``.uns['rank_genes_groups']``. Cluster labels are read from
            ``grn_adata3.obs.leiden_remap`` (module-level variable).
        keep_edges (dict): Nested regulon dict as returned by :func:`select_top_edges`,
            accessed via ``keep_edges['all'][cluster]['edges']``.
        n_top (int): Number of top-ranked genes per cluster to consider.
            Defaults to 50.

    Returns:
        dict: Mapping of source gene name to a square ``pd.DataFrame`` where
        both index and columns are cluster labels and values are Jaccard
        similarity scores. Only genes appearing in more than one cluster are
        included.
    """
    top_sources = {}
    top_counter = {}
    for ct in grn_adata3.obs.leiden_remap.unique():
        print(ct)
        try:
            bcrank = sc.get.rank_genes_groups_df(all_signatures, group=ct, key='wilcoxon')
            bcrank[['celltype', 'gene']] = bcrank['names'].str.rsplit('_', n=1, expand=True)
            topg = set(bcrank[0:n_top].gene)

            for g in topg:
                re = keep_edges['all'][ct]['edges']
                targets = list(re[re.source == g].target)
                if g in top_sources:
                    top_sources[g][ct] = targets
                    top_counter[g] += 1
                else:
                    top_sources[g] = {ct: targets}
                    top_counter[g] = 1
        except:
            pass

    # Dictionary to store the final DataFrames
    gene_matrices = {}

    for g, celltype_dict in top_sources.items():
        # Only process if the gene appears in more than 1 celltype
        if len(celltype_dict) < 2:
            continue

        results = []
        celltypes = sorted(celltype_dict.keys())

        for s1 in celltypes:
            set1 = set(celltype_dict[s1])
            for s2 in celltypes:
                set2 = set(celltype_dict[s2])

                # Calculate Jaccard
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                sim = intersection / union if union > 0 else 0

                results.append({
                    'ct1': s1,
                    'ct2': s2,
                    'jaccard': sim
                })

        # Convert to DataFrame and Pivot to Square Matrix
        df_long = pd.DataFrame(results)
        matrix = df_long.pivot(index='ct1', columns='ct2', values='jaccard')

        gene_matrices[g] = matrix

    return gene_matrices


def make_global_target_similarity_plot(gene_matrices):
    """Plot a heatmap of the mean pairwise Jaccard similarity across all source genes.

    Averages the per-source-gene Jaccard matrices from
    :func:`get_sourcewise_jaccard_regulons` into a single global similarity matrix,
    applies hierarchical clustering to both rows and columns, then renders an annotated
    seaborn heatmap with the colour bar repositioned to the lower-left corner.

    Args:
        gene_matrices (dict): Mapping of source gene name to a square Jaccard
            similarity ``pd.DataFrame`` as returned by
            :func:`get_sourcewise_jaccard_regulons`. An empty dict causes an
            early-exit message to be printed.

    Returns:
        None: Displays the plot via ``plt.show()``. No value is returned.
    """
    # 1. Stack all matrices and calculate the mean
    # We use .values to ensure we are averaging the numbers,
    # but keep the index/columns from one of the matrices.
    all_mats = list(gene_matrices.values())

    if len(all_mats) > 0:
        # Use reduce or concat to get the average
        global_matrix = pd.concat(all_mats).groupby(level=0).mean()
        # Ensure the columns are in the same order as the index for a perfect square
        global_matrix = global_matrix[global_matrix.index]
    else:
        print("No matrices found to average.")

    # Independent row and column clustering
    row_idx = hierarchy.leaves_list(hierarchy.linkage(pdist(global_matrix.fillna(0)), method='ward'))
    ordered_global = global_matrix.iloc[row_idx, row_idx]

    # 3. Plot
    fig, ax = plt.subplots(figsize=(7, 8))

    sns.heatmap(
        ordered_global,
        mask=(ordered_global == 0),
        cmap='YlGnBu',
        square=True,
        linewidths=.5,
        linecolor='#eeeeee',
        ax=ax,
        cbar_kws={"shrink": 0.2, "orientation": "horizontal", "label": "Mean Jaccard"},
        annot=False
    )

    # Move Legend to lower left
    cbar = ax.collections[0].colorbar
    cbar.ax.set_position([0.15, 0.05, 0.2, 0.015])

    # Formatting
    ax.tick_params(axis='both', which='major', pad=0.5, length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title('Global Target Similarity (Average across all Source Genes)', pad=25, fontweight='bold')

    plt.subplots_adjust(bottom=0.25, left=0.25)
    plt.show()
