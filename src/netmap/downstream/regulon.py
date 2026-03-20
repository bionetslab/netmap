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
from statsmodels.stats.multitest import multipletests
import networkx as nx
import requests
from pyvis.network import Network
import pyucell as ucell




from itertools import combinations
from collections import Counter


def select_top_edges(gene_inter_adata, adata, top_per_source=10, col_cluster='leiden_remap', min_reg_size=10, verbose=True, return_copy = False, tf_column=None):
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
        Column in obs defining clusters.grn_adata3.var

    Returns
    -------
    gene_inter_adata_filtered : AnnData
        Filtered AnnData containing top edges.
    reglon_sizes : list of int
        Sizes of regulatory regions per source.

    """

    min_edge_support = 0.5

    clusters = list(np.unique(gene_inter_adata.obs[col_cluster]))

    keep_edges_dict = {}


    for c in clusters:
        Keep_edges = []
        if verbose: print(f"Selecting targets for cluster: {c}")

        cells_c = gene_inter_adata.obs[col_cluster] == c
        gene_inter_adata.var['edge_support_c'] = (
            gene_inter_adata.layers['mask'][cells_c, :].mean(axis=0)
        )

        if tf_column is not None:
            tfs = gene_inter_adata.var[gene_inter_adata.var[tf_column]]['source'].unique()
            source_list =  set(gene_inter_adata.var["source"].unique()).intersection(set(tfs))
        
        else:
            source_list = gene_inter_adata.var["source"].unique()

        for source in source_list:

            df_targets = gene_inter_adata.var[
                    (gene_inter_adata.var['source'] == source) &
                    (gene_inter_adata.var['edge_support_c'] >= min_edge_support)]
            #print(gene_inter_adata[cells_c, gene_inter_adata[cells_c, df_targets.index].X.sum(axis = 0)].X.sum(axis = 0))
            
            df_targets['sum_of_edge'] = gene_inter_adata[cells_c, df_targets.index].X.sum(axis = 0)
            df_targets = df_targets.sort_values('sum_of_edge', ascending=False).head(top_per_source)

            if len(df_targets) >= min_reg_size:
                Keep_edges.extend(f"{source}_{t}" for t in df_targets['target'])

            keep_edges_dict[c] = Keep_edges
    keep_edges_dict = process_cell_edges(keep_edges_dict)
    return keep_edges_dict


def process_cell_edges(keep_edges):
    results = {'unique': {}, 'all': {}}
    all_cells = list(keep_edges.keys())

    
    def get_source_summary(edge_set):
        # Handles (source, target) tuples OR strings with a separator like '->'
        sources = []
        for e in edge_set:
            sources.append(e.split('_')[0])
        
        source_dict = dict(Counter(sources))
        sources = pd.DataFrame({'source' :source_dict.keys(), 'count': source_dict.values()}).sort_values('count', ascending=False)
        return sources

    # Calculate Uniques
    for cell in all_cells:
        others = set().union(*(set(keep_edges[c]) for c in all_cells if c != cell))
        unique = set(keep_edges[cell]) - others

        df = pd.DataFrame(
            [e.split('_', 1) for e in unique],
            columns=['source', 'target']
        )
        df_all  = pd.DataFrame(
            [e.split('_', 1) for e in set(keep_edges[cell])],
            columns=['source', 'target']
        )

        results['unique'][cell] = {
            'edges': df,
            'summary': get_source_summary(unique)
        }
        
        results['all'][cell] = {
            'edges': df_all,
            'summary': get_source_summary(set(keep_edges[cell]))
        }
        
    return results


def compute_signatures_UCell_scores(selected_edges, adata, key='unique') -> pd.DataFrame:
    """
    Filters gene signatures by cluster and computes UCell scores.

    Parameters
    ----------
    grn_adata : AnnData
        AnnData object containing GRN (gene regulatory network) information.
    adata : AnnData
        AnnData object containing gene expression counts in the 'counts' layer.

    Returns
    -------
    pd.DataFrame
        DataFrame with UCell scores merged with the 'spectral' cluster labels.
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


def aggregate_edges(selected_edges, grn_adata, key='unique') -> pd.DataFrame:
    """
    Filters gene signatures by cluster and computes UCell scores.

    Parameters
    ----------
    grn_adata : AnnData
        AnnData object containing GRN (gene regulatory network) information.
    adata : AnnData
        AnnData object containing gene expression counts in the 'counts' layer.

    Returns
    -------
    pd.DataFrame
        DataFrame with UCell scores merged with the 'spectral' cluster labels.
    """
    
    regulons = {}
    for ct in selected_edges[key]:
        print(ct)
        sign = selected_edges[key][ct]['edges'].groupby('source').apply(lambda x: (x['source'] + "_" + x['target']).tolist()).to_dict()
        for g in sign:
            regulons[f'{ct}_{g}'] = grn_adata[:, sign[g]].X.sum(axis = 1)/len(sign[g])
    regulons = pd.DataFrame(regulons)
    return regulons

