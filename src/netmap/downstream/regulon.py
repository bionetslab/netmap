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


from netmap.downstream.clustering import process, spectral_clustering, downstream_recipe
from netmap.downstream.edge_selection import add_top_edge_annotation_global


from itertools import combinations
from collections import Counter


def select_top_edges(gene_inter_adata, adata, top_per_source=10, col_cluster='leiden_remap', min_reg_size=10, verbose=True, return_copy = False, tf_column=None):

    min_edge_support = 0.5
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
            regulons[f'{ct}_{g}'] = grn_adata[:, sign[g]].X.mean(axis = 1)
    regulons = pd.DataFrame(regulons)
    return regulons
    


def make_cluster_regulon_dataframe(keep_edges):
    all_regulons = []
    for un in keep_edges:
        for clu in keep_edges[un] :
            df = keep_edges[un][clu]['edges']
            if df.shape[0]>0:
                df['cluster'] = un
                df['set_type'] = clu 
                all_regulons.append(df)
    all_regulons = pd.concat(all_regulons)
    return all_regulons



def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def get_sourcewise_jaccard_regulons(all_signatures, keep_edges, n_top = 50):
    top_sources = {}
    top_counter = {}
    for ct in grn_adata3.obs.leiden_remap.unique():
        print(ct)
        try:
            bcrank = sc.get.rank_genes_groups_df(all_signatures, group= ct, key='wilcoxon')
            bcrank[['celltype', 'gene']] = bcrank['names'].str.rsplit('_', n=1, expand=True)
            topg = set(bcrank[0:n_top].gene)

            for g in topg:
                re = keep_edges['all'][ct]['edges']
                targets = list(re[re.source == g].target)
                if g in top_sources:
                    top_sources[g][ct] = targets
                    top_counter[g]  +=1
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