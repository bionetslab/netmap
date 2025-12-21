# Standard library imports
import gc
import json
import logging
import os
import sys
import time
import warnings
from functools import reduce
from itertools import chain, product
from typing import Tuple, List, Dict, Optional
from anndata import AnnData
from itertools import combinations


# Third-party imports
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as scs
from anndata import AnnData
import seaborn as sns
import scipy.sparse as sp
import anndata
from scipy.stats import pearsonr, ranksums
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import contingency_matrix
from statsmodels.stats.multitest import multipletests
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from anndata import AnnData
import importlib
import json
import os
from turtle import shape
from typing import List, Optional, Tuple, Union

import networkx as nx
import pandas as pd
import requests
from pandas import DataFrame
from pyvis.network import Network
from anndata import AnnData
import importlib
import json
import os
from typing import List, Optional, Tuple, Union

import networkx as nx
import pandas as pd
from pandas import DataFrame
import pyucell as ucell
from pyvis.network import Network


#import omnipath as op

# Miscellaneous
warnings.filterwarnings("ignore")

from netmap.downstream.clustering import process, spectral_clustering, downstream_recipe
from netmap.downstream.edge_selection import add_top_edge_annotation_global


    


def filter_clusters_by_cell_count(grn_adata: AnnData, metric_tag: float, top_fraction: float) -> Tuple[Optional[Dict[str, float]], AnnData]:
    """
    Filter features (genes/edges) based on cell count differences between two clusters,
    selecting the top fraction of features for each cluster.

    Parameters
    ----------
    grn_adata : AnnData
        AnnData object containing feature metadata in `.var`.
        Must include columns `cell_count_<metric_tag>_0` and `cell_count_<metric_tag>_1`.
    metric_tag : str
        Identifier (suffix) for the cell count metric, e.g. '0.95' → columns
        `cell_count_0.95_0` and `cell_count_0.95_1`.
    top_fraction : float
        Fraction (0–1) of top features to retain per cluster.
        For example, 0.2 retains the top 20% of features.

    Returns
    -------
    filtered_adata : AnnData
        Subset of the input AnnData object containing only the top-selected features.
    """

    # --- Validate inputs ---
    if not (0 < top_fraction <= 1):
        raise ValueError("`top_fraction` must be between 0 and 1.")

    col_0 = f"cell_count_{metric_tag}_0"
    col_1 = f"cell_count_{metric_tag}_1"

    var_df = grn_adata.var
    if col_0 not in var_df.columns or col_1 not in var_df.columns:
        raise KeyError(f"Missing expected columns '{col_0}' or '{col_1}' in grn_adata.var.")

    threshold_n = max(1, int(len(var_df) * top_fraction))

    top_cluster0 = var_df.nlargest(threshold_n, [col_0, col_1])
    top_cluster1 = var_df.nlargest(threshold_n, [col_1, col_0])

    unique_indices = np.unique(np.concatenate([top_cluster0.index.values, top_cluster1.index.values]))
    filtered_adata = grn_adata[:, unique_indices]

    return filtered_adata


def get_top_targets(gene_inter_adata, adata, top_per_source=750, col_cluster='spectral', min_reg_size=100, verbose=True):
    """
    Selects top gene targets per source from a clustered gene interaction AnnData.

    Parameters
    ----------
    gene_inter_adata : AnnData
        Gene interaction AnnData with `var` containing 'source' and 'target'.
    adata : AnnData
        Expression AnnData for ranking genes.
    top_per_source : int, default=750
        Number of top targets to select per source.
    col_cluster : str, default='spectral'
        Column in obs defining clusters.

    Returns
    -------
    gene_inter_adata_filtered : AnnData
        Filtered AnnData containing top edges.
    reglon_sizes : list of int
        Sizes of regulatory regions per source.

    """

    if verbose: print(f"Initial shape: {gene_inter_adata.shape}")

    # Rank genes per cluster
    sc.tl.rank_genes_groups(adata, groupby=col_cluster, method='t-test')
    clusters = np.unique(gene_inter_adata.obs[col_cluster])

    # Merge ranking data across clusters
    rank_dfs = []
    for c in clusters:
        if verbose: print(f"Ranking cluster: {c}")
        df = sc.get.rank_genes_groups_df(adata, group=str(c))
        df = df.sort_values('scores', ascending=False)
        df[f"rank_{c}"] = np.arange(1, len(df) + 1)
        rank_dfs.append(df[['names', f'rank_{c}']])
    df_rank = reduce(lambda l, r: pd.merge(l, r, on='names', how='inner'), rank_dfs)

    # Compute differences per cluster
    Keep_edges, reglon_sizes = [], []
    for c in clusters:
        if verbose: print(f"Selecting targets for cluster: {c}")
        df_rank_c = df_rank.copy()
        rank_cols = [col for col in df_rank.columns if col != 'names']
        rank_cols.remove(f"rank_{c}")
        df_rank_c['avg'] = df_rank_c[rank_cols].mean(axis=1)
        df_rank_c['diff'] = (df_rank_c[f"rank_{c}"] - df_rank_c['avg']).abs()
        df_rank_c = df_rank_c.sort_values('diff', ascending=False)

        # Process sources
        for source in gene_inter_adata.var["source"].unique():
            tf_rank = df_rank_c.loc[df_rank_c['names'] == source, 'diff'].values[0]
            df_targets = (
                gene_inter_adata.var[gene_inter_adata.var['source'] == source]
                .merge(df_rank_c[['names', 'diff']], left_on='target', right_on='names', how='left')
            )
            df_targets['rank_distance'] = (df_targets['diff'] - tf_rank).abs()
            df_targets = df_targets.sort_values('rank_distance').head(top_per_source)

            reglon_sizes.append(len(df_targets))
            if len(df_targets) >= min_reg_size:
                Keep_edges.extend(f"{source}_{t}" for t in df_targets['target'])

    # Deduplicate and subset
    Keep_edges = list(set(chain.from_iterable([[k] for k in Keep_edges])))
    if verbose:
        print(f"Edges kept: {len(Keep_edges)} unique")

    gene_inter_adata_filtered = gene_inter_adata[:, gene_inter_adata.var.index.isin(Keep_edges)].copy()
    if verbose:
        print(f"Filtered shape: {gene_inter_adata_filtered.shape}")
        print(f"Unique sources: {len(gene_inter_adata.var['source'].unique())}")
        print(f"After selecting top_per_source={top_per_source}")

    return gene_inter_adata_filtered, reglon_sizes


def filter_signatures_by_Ucell(grn_adata, adata, ncores: int = 100) -> pd.DataFrame:
    """
    Filters gene signatures by cluster and computes UCell scores.

    Parameters
    ----------
    grn_adata : AnnData
        AnnData object containing GRN (gene regulatory network) information.
    adata : AnnData
        AnnData object containing gene expression counts in the 'counts' layer.
    ncores : int, optional
        Number of cores to use for parallel computation, by default 100.

    Returns
    -------
    pd.DataFrame
        DataFrame with UCell scores merged with the 'spectral' cluster labels.
    """
    
    signatures  = grn_adata.var.groupby('source')['target'].apply(list).to_dict()
    ucell.compute_ucell_scores(adata, signatures=signatures)
    data_ucell = adata.obs.filter(like='_UCell')
    return data_ucell



def filter_grn_by_top_signatures(data_ucell: pd.DataFrame, grn_adata: AnnData, keep_top_ranked: int = 100, filter_by: str = "z_score") -> Tuple[Optional[AnnData], List[str]]:
    """
    Filters a GRN (Gene Regulatory Network) AnnData object to keep only the edges
    corresponding to the top-ranked signatures per cluster based on UCell scores.

    Parameters
    ----------
    data_ucell : pd.DataFrame
        DataFrame containing UCell scores for each signature across cells, with a 
        'spectral' column indicating cluster assignments.
    grn_adata : AnnData
        The GRN AnnData object to filter, containing edge metadata in `.var`.
    keep_top_ranked : int, optional
        Number of top signatures to retain per cluster (default is 100).
    filter_by : str, optional
        Metric used to rank signatures. Options include "P_Value", "Adjusted_P_Value", or "z_score" (default is "P_Value").

    Returns
    -------
    Tuple[Optional[AnnData], List[str]]
        - Filtered AnnData object containing only edges corresponding to top-ranked signatures.
        - List of unique top source names used to filter the GRN.

    Notes
    -----
    - For each cluster, signatures are ranked using the chosen metric.
    - When `filter_by='z_score'`, a combined score of -log10(P_Value) and z_score is used.
    - The function returns `None` and an empty list if the input GRN has no variables.
    """
    if grn_adata.var.empty:
        return None, []

    clusters = grn_adata.obs['spectral']
    ucell_scores = data_ucell

    signature_bases = [sig.split("_UCell")[0] for sig in ucell_scores.columns]

    cluster_top_sources = []

    for cluster in clusters.unique():
        mask = clusters.values == cluster
        cluster_cells = ucell_scores.loc[mask]
        other_cells = ucell_scores.loc[~mask]

        stats = [ranksums(cluster_cells[sig], other_cells[sig], alternative='greater') 
                 for sig in ucell_scores.columns]
        stat_vals, p_values = zip(*stats)
        mean_diffs = (cluster_cells.mean() - other_cells.mean()).abs().tolist()
        adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]

        results_df = pd.DataFrame({
            'Signature': ucell_scores.columns,
            'P_Value': p_values,
            'Stat': stat_vals,
            'Adjusted_P_Value': adjusted_p_values,
            'Mean_Difference': mean_diffs
        })

        if filter_by == 'z_score':
            results_df['z_score'] = (results_df['Stat'] - results_df['Stat'].mean()) / results_df['Stat'].std()
            results_df['combined_score'] = -np.log10(results_df['P_Value'] + 1e-300) + results_df['z_score']
            top_n = results_df.nlargest(keep_top_ranked, 'combined_score')
        else:
            ascending = filter_by in ['P_Value', 'Adjusted_P_Value']
            top_n = results_df.sort_values(filter_by, ascending=ascending).head(keep_top_ranked)

        top_sources = [signature_bases[ucell_scores.columns.get_loc(sig)] for sig in top_n["Signature"]]
        cluster_top_sources.extend(top_sources)

    top_sources_list = list(set(cluster_top_sources))
    grn_adata_filtered = grn_adata[:, grn_adata.var["source"].isin(top_sources_list)].copy()

    return grn_adata_filtered, top_sources_list


# def filter_grn_by_top_signatures(data_ucell: pd.DataFrame, grn_adata: AnnData, keep_top_ranked: int = 100) -> Tuple[Optional[AnnData], List[str]]:
    
#     if grn_adata.var.empty:
#         return None, []

#     df = data_ucell.copy()
#     features = [c for c in df.columns if c.endswith('_UCell')]
#     clusters = sorted(df['spectral'].unique())
#     all_results = []

#     for cl in clusters:
#         g1, g2 = df[df['spectral'] == cl], df[df['spectral'] != cl]
#         res = []
#         for f in features:
#             try:
#                 s, p = mannwhitneyu(g1[f], g2[f], alternative='two-sided')
#                 res.append({'cluster': cl, 'gene_set': f, 'mean_diff': g1[f].mean() - g2[f].mean(), 'pval': p})
#             except: 
#                 res.append({'cluster': cl, 'gene_set': f, 'mean_diff': 0, 'pval': 1})
#         res = pd.DataFrame(res)
#         res['padj'] = multipletests(res['pval'], method='fdr_bh')[1]
#         all_results.append(res.sort_values(['padj','mean_diff'], ascending=[True, False]).head(keep_top_ranked))

#     combined = pd.concat(all_results, ignore_index=True)
#     top_sources = [s.split("_UCell")[0] for s in list(combined["gene_set"])]
#     top_sources_list = list(set(top_sources))

#     grn_adata_filtered = grn_adata[:, grn_adata.var["source"].isin(top_sources_list)].copy()
    
#     return grn_adata_filtered, top_sources_list



def plot_shared_targets_heatmap(grn_adata, genes=None, figsize=(6, 6), cmap='RdBu_r',
                                metric='euclidean', method='average', title='Clustered Heatmap of Shared Targets'):
    """
    Create a clustered heatmap showing shared target counts between sources.

    Parameters
    ----------
    grn_adata : AnnData
        An AnnData object where the gene regulatory network info is in `grn_adata.var`.
        Must contain columns 'source' and 'target'.
    genes : list, optional
        A list of source genes to include. If None, all sources are used.
    figsize : tuple, optional
        Size of the heatmap figure.
    cmap : str, optional
        Colormap for the heatmap.
    metric : str, optional
        Distance metric for clustering (passed to seaborn.clustermap).
    method : str, optional
        Linkage method for clustering (passed to seaborn.clustermap).
    title : str, optional
        Title for the heatmap.

    Returns
    -------
    shared_target_matrix : pd.DataFrame
        A symmetric DataFrame of shared target counts between sources.
    """
    # Optionally subset by selected genes
    if genes is not None:
        grn_adata = grn_adata[:, grn_adata.var['source'].isin(genes)].copy()

    # Load the data
    df = grn_adata.var.copy()

    # Build mapping from source to set of targets
    source_targets = df.groupby('source')['target'].apply(set)

    # If fewer than 2 sources, skip
    if len(source_targets) < 2:
        raise ValueError("Need at least two sources for comparison.")

    # Create similarity matrix
    sources = source_targets.index.tolist()
    shared_target_matrix = pd.DataFrame(0, index=sources, columns=sources, dtype=int)

    # Count shared targets for each source pair
    for src1, src2 in combinations(sources, 2):
        shared = len(source_targets[src1] & source_targets[src2])
        shared_target_matrix.loc[src1, src2] = shared
        shared_target_matrix.loc[src2, src1] = shared

    # Fill diagonal with number of targets per source
    for src in sources:
        shared_target_matrix.loc[src, src] = len(source_targets[src])

    # Plot clustered heatmap
    sns.clustermap(shared_target_matrix, cmap=cmap, figsize=figsize,
                   metric=metric, method=method)
    plt.suptitle(title, y=1.05)
    plt.show()

    #return shared_target_matrix

#**********************************************************************



def compute_edge_overlaps_simple(grn_adata: AnnData, net_list: List[Tuple[str, pd.DataFrame]]) -> Dict[str, float]:
    """
    Compute the percentage of overlapping edges between a GRN and multiple reference networks.

    Parameters
    ----------
    grn_adata : AnnData
        AnnData object with edges in `grn_adata.var`, containing 'source' and 'target' columns.
    net_list : List[Tuple[str, pd.DataFrame]]
        List of tuples (network_name, network_dataframe), where each dataframe
        must include 'source' and 'target' columns.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each network name (with index suffix) to its overlap percentage.
        Example: {'STRING__0': 0.32, 'BioGRID__1': 0.45}
    """
    # --- Basic input checks ---
    if not all(col in grn_adata.var.columns for col in ["source", "target"]):
        raise KeyError("`grn_adata.var` must contain 'source' and 'target' columns.")

    overlap_perc = {}

    for idx, (net_name, net_df) in enumerate(net_list):
        if not all(col in net_df.columns for col in ["source", "target"]):
            raise KeyError(f"Network '{net_name}' missing 'source' or 'target' columns.")

        overlap_count = pd.merge(grn_adata.var[["source", "target"]], net_df[["source", "target"]], on=["source", "target"], how="inner").shape[0]
        net_size = net_df.shape[0]
        overlap_perc[f"{net_name}__{idx}"] = (overlap_count / net_size)*100 if net_size > 0 else 0.0

    return overlap_perc



def filter_clusters_by_cell_count(grn_adata: AnnData, metric_tag: float, top_fraction: float) -> Tuple[Optional[Dict[str, float]], AnnData]:
    """
    Filter features (genes/edges) based on cell count differences between two clusters,
    selecting the top fraction of features for each cluster.

    Parameters
    ----------
    grn_adata : AnnData
        AnnData object containing feature metadata in `.var`.
        Must include columns `cell_count_<metric_tag>_0` and `cell_count_<metric_tag>_1`.
    metric_tag : str
        Identifier (suffix) for the cell count metric, e.g. '0.95' → columns
        `cell_count_0.95_0` and `cell_count_0.95_1`.
    top_fraction : float
        Fraction (0–1) of top features to retain per cluster.
        For example, 0.2 retains the top 20% of features.

    Returns
    -------
    filtered_adata : AnnData
        Subset of the input AnnData object containing only the top-selected features.
    """

    # --- Validate inputs ---
    if not (0 < top_fraction <= 1):
        raise ValueError("`top_fraction` must be between 0 and 1.")

    col_0 = f"cell_count_{metric_tag}_0"
    col_1 = f"cell_count_{metric_tag}_1"

    var_df = grn_adata.var
    if col_0 not in var_df.columns or col_1 not in var_df.columns:
        raise KeyError(f"Missing expected columns '{col_0}' or '{col_1}' in grn_adata.var.")

    threshold_n = max(1, int(len(var_df) * top_fraction))

    top_cluster0 = var_df.nlargest(threshold_n, [col_0, col_1])
    top_cluster1 = var_df.nlargest(threshold_n, [col_1, col_0])

    unique_indices = np.unique(np.concatenate([top_cluster0.index.values, top_cluster1.index.values]))
    filtered_adata = grn_adata[:, unique_indices]

    return filtered_adata



def create_regulon_activity_adata(grn_adata: AnnData, data_ucell: pd.DataFrame, top_sources_list: List[str]) -> AnnData:
    """
    Creates an AnnData object with regulon activity scores based on top GRN sources.

    Parameters
    ----------
    grn_adata : AnnData
        Original GRN AnnData object, used for alignment of obs and embeddings.
    data_ucell : pd.DataFrame
        DataFrame containing UCell scores for all signatures.
    top_sources_list : List[str]
        List of top source names to include as regulons.

    Returns
    -------
    AnnData
        New AnnData object with regulon activity, aligned to grn_adata observations.
    """
    
    # Standardize column names
    data_ucell = data_ucell.copy()
    data_ucell.columns = [col.split("_")[0] + "_regulon" for col in data_ucell.columns]

    # Keep only top regulons that exist in data_ucell
    expected_cols = [f"{s}_regulon" for s in top_sources_list if f"{s}_regulon" in data_ucell.columns]
    data_ucell = data_ucell[expected_cols]

    # Create AnnData object
    adata_regulon = ad.AnnData(
        X=data_ucell.values,
        obs=grn_adata.obs.copy(),
        var=pd.DataFrame(index=expected_cols)
    )

    # Align embeddings if available
    if 'X_umap' in grn_adata.obsm:
        adata_regulon.obsm['X_umap'] = grn_adata.obsm['X_umap'].copy()

    return adata_regulon




def plot_reg(grn_adata: AnnData, regulon: List, name="network", layout: Optional[str]="hierarchical", out_path="network_plots/"):
    
    # Make all genes uppercase
    #df = df.applymap(lambda s: s.upper() if type(s) == str else s)
    
    #shap1 = df.shape[0]
    #df = df[df["occurrence(pct)"]>=occurrence_pct]
    #shap2 = df.shape[0]
    #print(f"\n Out of {shap1} edges, {shap2} edges satisfying occurrence threshold {occurrence_pct}% where kept \n")

    df = grn_adata.var[grn_adata.var["source"].isin(regulon)]
    df = df[df["source"].isin(regulon)]
    
    TFs = list(set(list(df["source"])))
    print(TFs)
    print(df.shape)

    intersection = list(set(regulon) & set(TFs))
    if len(intersection)<1:
        raise Exception(f"{regulon} is not a regulator please chose from: {TFs}") 

    # if regulon != "all":
    #     if regulon in TFs:
    #         print(f"plottin the {regulon} network ...")
    #         df = df[df["TF"]==regulon]
    #     else:
    #         raise Exception(f"{regulon} is not a regulator please chose from: {TFs}") 

    graph = nx.from_pandas_edgelist(df,"source","target")
    graph.remove_edges_from(nx.selfloop_edges(graph))
    #print(layout)
    network = Network(height='500px', width='800px', directed=True, notebook=True, cdn_resources='in_line')
    network.from_nx(graph)
    
    net_genes = [x["id"] for x in network.nodes]
    # drug interaction
    # if drug_interaction:
    #     drug_interactions = DrugInteractions.get_drug_interactions(net_genes, algorithm)
    #     for inter in drug_interactions:
    #         network.add_node(inter[0], shape='star', color='darkgreen')
    #         network.add_edge(inter[0],inter[1])

    for node in network.nodes:
        if node["label"] in TFs:
            node['shape'] = "triangleDown"
            node['color'] = "orangered"
            node['size'] = 16
        else:
            node['size'] = 14
    for edge in network.edges:
        edge['color'] = "black"
        
    
    network.set_edge_smooth('cubicBezier')
    print(f"path: {out_path}networks/{name}.html")
    return network.show(f"{out_path}networks/{name}.html")



def curate_network(grn_adata_, organism="human", tfs=None):
    """
    Load CollecTRI data for a specific organism, format it, and curate the network data
    based on the provided gene regulatory network (GRN) data.

    Parameters:
    -----------
    grn_adata_ : AnnData
        Annotated data object containing the gene regulatory network (GRN) data.
    
    organism : str, optional (default="human")
        The organism to retrieve data for from CollecTRI.

    tfs : list of str or None, optional (default=None)
        A list of transcription factors (source genes) to retain. If None, all TFs are used.

    Returns:
    --------
    net_df_curated : pandas.DataFrame
        The curated network DataFrame, containing the merged GRN data and CollecTRI interaction data.
    """
    
    # Load CollecTRI data
    collectri_df = op.interactions.CollecTRI.get(genesymbols=True, organism=organism)
    
    # Format to required columns
    collectri_df = collectri_df[['source_genesymbol', 'target_genesymbol', 'is_stimulation', 'is_inhibition']]

    # Optionally filter by specified TFs
    if tfs is not None:
        collectri_df = collectri_df[collectri_df['source_genesymbol'].isin(tfs)]

    # Determine interaction mode
    def get_mode(row):
        if row['is_stimulation']:
            return 'Activation'
        elif row['is_inhibition']:
            return 'Repression'
        else:
            return 'Unknown'

    collectri_df['mode'] = collectri_df.apply(get_mode, axis=1)

    # Merge with GRN data
    net_df_curated = pd.merge(grn_adata_.var, collectri_df, 
                              left_on=['source', 'target'], 
                              right_on=['source_genesymbol', 'target_genesymbol'])

    return net_df_curated

def plot_reg_curated(net_df: DataFrame, regulon: List, name="network", layout: Optional[str]="hierarchical", out_path="network_plots/"):
    
    # Filter network for selected regulon
    df = net_df
    df = df[df["source"].isin(regulon)]
    #print(df)
    
    TFs = list(set(list(df["source"])))
    #print(TFs)
    #print(df.shape)

    intersection = list(set(regulon) & set(TFs))
    if len(intersection) < 1:
        raise Exception(f"{regulon} is not a regulator please choose from: {TFs}") 

    # Build directed graph
    graph = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.DiGraph())
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Dynamic network with PyVis
    network = Network(height='600px', width='100%', directed=True, notebook=True, cdn_resources='in_line')
    network.from_nx(graph)
    
    net_genes = [x["id"] for x in network.nodes]
    
    # Edge color mapping
    edge_colors = {'Activation': 'green', 'Repression': 'red', 'Unknown': 'gray'}

    # Style nodes
    for node in network.nodes:
        if node["label"] in TFs:
            node['shape'] = "triangleDown"
            node['color'] = "orangered"
            node['size'] = 18
        else:
            node['shape'] = "dot"
            node['color'] = "lightblue"
            node['size'] = 14

    # Style edges
    for edge in network.edges:
        edge_data = graph.get_edge_data(edge['from'], edge['to'])
        mode = edge_data.get('mode', 'Unknown') if edge_data else 'Unknown'
        edge['color'] = edge_colors.get(mode, 'gray')
        edge['width'] = 2

    # Smooth curved arrows
    network.set_edge_smooth('cubicBezier')

    # ✅ Correct JSON layout options (no JavaScript)
    if layout == "hierarchical":
        network.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "hubsize",
              "shakeTowards": "roots"
            }
          },
          "physics": {
            "hierarchicalRepulsion": {
              "nodeDistance": 150
            }
          }
        }
        """)
    else:
        # Use Barnes-Hut layout
        network.barnes_hut()

    # Save HTML
    os.makedirs("{out_path}networks", exist_ok=True)
    output_path = f"{out_path}networks/{name}.html"
    network.show(output_path)
    print(f"✅ Interactive PyVis network saved to {output_path}")

    return network.show(f"{out_path}networks/{name}.html")