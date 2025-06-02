import numpy as np
import scanpy as sc
import logging
from typing import Optional, Dict
import pandas as pd
import numpy
#import omnipath as op
import scipy.sparse as sp
import rpy2.robjects as ro
import rpy2.robjects.pandas2ri as pandas2ri
import pandas as pd
pandas2ri.activate()
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from itertools import chain
import scanpy as sc

import pandas as pd
import numpy as np
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad

logging.basicConfig(level=logging.INFO)

def downstream_recipe(adata, min_perc: float, n_clusters: int):

    """
    Performs a downstream analysis on an AnnData object, including filtering, PCA, clustering, and UMAP.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix.
    min_perc : float
        Minimum percentage of cells an edge must be expressed in to be retained.
    n_clusters : int
        Desired number of clusters.
    config : dict, optional
        Dictionary containing 'n_neighbors', 'leiden_resolution', 'n_components', 'knn_neighbors'. Defaults used if not provided.
    
    Returns:
    --------
    AnnData
        Updated AnnData object with PCA, clustering, and UMAP embeddings.
    """

    default_config = {'n_neighbors': 30, 'n_components': 30, 'knn_neighbors': 30}
    config = default_config

    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=int(adata.n_obs * (min_perc / 100)))
    n_genes_after = adata.n_vars
    print(f"Removed {n_genes_before - n_genes_after} edges (kept {n_genes_after} edges).")

    sc.tl.pca(adata, svd_solver='randomized', zero_center=False)

    sc.pp.neighbors(adata, n_neighbors=config['knn_neighbors'])

    cluster_found = False
    for res in np.linspace(0.05, 1.5, 30):
        sc.tl.leiden(adata, resolution=res)
        n_unique = len(np.unique(adata.obs['leiden']))
        if n_unique == n_clusters:
            cluster_found = True
            break

    if not cluster_found:
        raise ValueError(f"Could not find exactly {n_clusters} clusters. Last resolution produced {n_unique} clusters.")

    sc.tl.umap(adata, n_components=config['n_components'])

    return adata


def filter_low_count_edges(adata, min_mean_count=0.005):
    """
    Filters edges with mean counts below a given threshold after calculating QC metrics.

    Parameters:
    -----------
    adata : AnnData
        The input AnnData object.
    min_mean_count : float
        Minimum mean count threshold to retain edges.

    Returns:
    --------
    AnnData
        Filtered AnnData object.
    """

    sc.pp.calculate_qc_metrics(adata, inplace=True)
    n_genes_before = adata.n_vars
    adata = adata[:, adata.var['mean_counts'] > min_mean_count]
    n_genes_after = adata.n_vars

    print(f"Filtered out {n_genes_before - n_genes_after} genes (kept {n_genes_after}).")
    return adata




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


def top_genes_per_source_selection(adata, gene, cluster, top_per_source):

    adata = adata[adata.obs["leiden"] == cluster]
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    g = gene
    g_idx = adata.var_names.get_loc(g)
    corr = np.corrcoef(X[:, g_idx], X.T)[0, 1:]
    corr_series = pd.Series(corr, index=adata.var_names).drop(g)
    return [f"{g}_{t}" for t in list(corr_series.abs().sort_values(ascending=False).head(top_per_source).index)]

def top_de_genes_from_list(adata, source_gene, gene_list, cluster, top_n):

    ranked_genes = adata.uns['rank_genes_groups']['names'][cluster]
    selected_genes = [gene for gene in ranked_genes if gene in gene_list]
    keep_l = selected_genes[:top_n]
    return [f"{source_gene}_{t}" for t in keep_l]

def filter_signatures_by_cluster(adata, grn_adata, cluster, top_per_source, min_regulon_size, ncores=100):

    """
    Group regulons by their source transcription factors and rank them based on differential activity between the current cluster and all other clusters. 
    The ranking reflects the magnitude and significance of activity shifts, prioritizing source genes whose downstream targets show distinct regulatory behavior in the specified cluster.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing the single-cell gene expression matrix and metadata.
        
    grn_adata : AnnData
        Annotated data object containing the gene regulatory network (GRN) data, which includes
        the edges between source genes and their target genes.
        
    cluster : str
        The cluster name (or label) within the Leiden clustering for which to select top genes and perform analysis.
        
    top_per_source : int
        The number of top genes to select per source gene based on DGE ranking.
        
    min_regulon_size : int
        The minimum size of a regulon (number of target genes for a source gene) for the source gene to be considered.
        
    ncores : int, optional, default=100
        The number of CPU cores to use for parallelization when calculating UCell scores.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the UCell scoring, including the P-values, statistics, 
        adjusted P-values, and mean differences for the gene signatures.
        
    all_edges : pandas.DataFrame
        A DataFrame containing the filtered edges (source-target relationships) after applying 
        the minimum regulon size filter.
        
    data : pandas.DataFrame
        A DataFrame with UCell scores merged with cluster information from adata.obs.
        
    Notes:
    ------
    - The function relies on the UCell package in R to compute the gene signature scores for each source gene.
    - It performs a statistical comparison between the selected cluster and the rest of the clusters using the 
      Wilcoxon rank-sum test to identify differentially expressed genes.
    - Adjusts p-values using the Benjamini-Hochberg method to control for false discovery rate.
    - Uses UCell scores to rank gene signatures and computes statistical differences between clusters.
    """
    

    df = grn_adata.var
    group_size = df.groupby('source').size().sort_values(ascending=False)
    valid_sources = group_size[group_size > min_regulon_size].index
    df_gene = df[df['source'].isin(valid_sources)]
    print(f"{valid_sources.shape[0]} source based groups are found")
    print(f'{df_gene.shape[0]} edges are found')
    print("\n")    

    if "leiden" in adata.obs:
        del adata.obs["leiden"]
    adata.obs = adata.obs.merge(grn_adata.obs[["leiden"]], left_index=True, right_index=True)

    if sp.issparse(adata.X):  
        count_matrix = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    else:
        count_matrix = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    #print(f"N sources: {len(list(df_gene['source'].unique()))}")

    # Keep_this_edges = []
    # for S in list(df_gene["source"].unique()):
    #     Keep_this_edges.append(top_genes_per_source_selection(adata, S, cluster, top_per_source))
    # Keep_this_edges = list(chain.from_iterable(Keep_this_edges))
    # print(f"length Keep_this_edges: {len(list(set(Keep_this_edges)))}")

    sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')
    Keep_this_edges = []
    for S in list(df_gene["source"].unique()):

        keep_ = top_de_genes_from_list(adata, S, df_gene[df_gene["source"] == S]["target"].values, cluster=cluster, top_n=top_per_source)
        Keep_this_edges.append(keep_)

    Keep_this_edges = list(chain.from_iterable(Keep_this_edges))
    #print(f"length Keep_this_edges: {len(list(set(Keep_this_edges)))}")


    #print("\n")
    #print(f"df_gene shape: {df_gene.shape}")
    all_edges = df_gene[df_gene.index.isin(Keep_this_edges)]
    #print(f"all_edges shape: {all_edges.shape}")
    #all_edges['edge'] = all_edges.index
    #print(all_edges["source"] .value_counts())
    #print(f"all_edges shape: {all_edges.shape}")
    #print(all_edges.head(4))
    #print("\n")
    
    #print("In R")
    ncores = ncores
    ro.r('library(UCell)')
    ro.r('library(Matrix)')
    ro.globalenv['count_matrix'] = pandas2ri.py2rpy(count_matrix)  
    ro.globalenv['all_edges'] = pandas2ri.py2rpy(all_edges) 
    ro.globalenv['ncores'] = ncores
    
    # R code
    ro.r('''
    mat <- as.matrix(t(count_matrix))
    sparse_mat <- as(mat, "CsparseMatrix")
    
    #print(dim(sparse_mat))
    #print(class(sparse_mat))
    
    df <- all_edges
    
    df$source <- as.character(df$source)
    df$target <- as.character(df$target)
    
    # Use split() and lapply to create the result list, grouping by 'source'
    result_list <- lapply(split(df, df$source), function(group) {
      c(group$source[1], group$target)  # Add the source and all its targets
    })
    
    # Print the first few elements of the result list
    # print(length(result_list))
    
    scores <- ScoreSignatures_UCell(sparse_mat, features=result_list, ncores=ncores)
    scores_df <- as.data.frame(scores)
    
    print("dimensions of the resulting scores")
    print(dim(scores_df))
    ''')
    
    scores_df_r = ro.r['scores_df']
    scores_df = pandas2ri.rpy2py(scores_df_r)
    print("\n")

    data = scores_df.merge(adata.obs[["leiden"]], left_index=True, right_index=True)
    #print(data.shape)
    
    # Extract UCell scores and clusters
    ucell_scores = data.drop(columns=['leiden'])
    clusters = data['leiden']
    
    # Split data by clusters
    cluster1_cells = ucell_scores[clusters == cluster]
    cluster2_cells = ucell_scores[clusters != cluster]
    
    # Perform statistical tests and calculate mean differences
    p_values, mean_diffs, stat_ = [], [], []
    for signature in ucell_scores.columns:
        cluster1_scores = cluster1_cells[signature]
        cluster2_scores = cluster2_cells[signature]
        stat, p_val = ranksums(cluster1_scores, cluster2_scores, alternative='greater')
        mean_diffs.append(abs(cluster1_scores.mean() - cluster2_scores.mean()))
        p_values.append(p_val)
        stat_.append(stat)
    
    # Adjust p-values for multiple testing
    adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
    
    # Create result DataFrame and rank the results
    results_df = pd.DataFrame({
        'Signature': ucell_scores.columns,
        'P_Value': p_values,
        'Stat': stat_,
        'Adjusted_P_Value': adjusted_p_values,
        'Mean_Difference': mean_diffs
    })
    
    # Rank by p-value first (lower is better)
    #results_df['P_Value_Rank'] = results_df['Adjusted_P_Value'].rank(method='min')
    
    # Rank by mean difference (higher is better)
    #results_df['Mean_Difference_Rank'] = results_df['Mean_Difference'].rank(ascending=True, method='min')
    
    # Combine the ranks into a composite rank
    #results_df['Composite_Rank'] = results_df['P_Value_Rank'] + results_df['Mean_Difference_Rank']
    
    # Sort by composite rank (lower is better)
    #results_df['Rank'] = results_df['Composite_Rank'].rank(method='min')
    
    # Sort the results by rank and display top 10 ranked signatures
    #results_df_sorted = results_df.sort_values('Rank')
    
    # Display top 10 ranked signatures
    #print(results_df_sorted.shape)
    #results_df_sorted.head(2)
    
    return results_df, all_edges, data



def filter_grn_by_top_signatures(grn_adata, results_df, all_edges, keep_top_ranked=100):
    """
    Filter the top N ranked signatures based on P_Value.

    Parameters:
    -----------
    grn_adata : AnnData
        Full GRN AnnData object with `source` and `target` in .var and `leiden` in .obs.

    results_df : pd.DataFrame
        DataFrame containing ranked signatures with a 'P_Value' column and 'Signature' column.

    all_edges : pd.DataFrame
        DataFrame with 'source' and 'target' columns representing known or scored edges.

    keep_top_ranked : int, default=100
        Number of top-ranked signatures to use for filtering.

    Returns:
    --------
    grn_adata_filtered : AnnData
        Filtered AnnData object containing only top-ranked and matched edges.
    """

    # Construct "Edges" identifier
    all_edges["Edges"] = all_edges["source"].astype(str) + "_" + all_edges["target"].astype(str)

    # Get top N signatures based on smallest p-values
    top_n = results_df.nsmallest(keep_top_ranked, 'P_Value')
    print(f"top_n.shape: {top_n.shape}")

    # Extract unique signature source names (strip suffix)
    top_n_l_ = list(top_n["Signature"])
    top_n_l = [x.split("_UCell")[0] for x in top_n_l_]

    # Subset GRN to relevant sources
    grn_adata__ = grn_adata[:, grn_adata.var["source"].isin(top_n_l)].copy()

    # Create minimal AnnData object with relevant columns
    grn_adata_ = sc.AnnData(
        X=grn_adata__.X,
        obs=grn_adata__.obs[['leiden']],
        var=grn_adata__.var[['source', 'target']]
    )

    # Define new var_names as "source_target"
    grn_adata_.var_names = grn_adata_.var['source'].astype(str) + '_' + grn_adata_.var['target'].astype(str)

    # Keep only matching edges
    edge_set = set(all_edges["Edges"])
    to_keep_e = grn_adata_.var_names[grn_adata_.var_names.isin(edge_set)]
    print(f"Filtered edge count: {len(to_keep_e)}")

    # Subset again to keep only matched edges
    grn_adata_filtered = grn_adata_[:, to_keep_e].copy()
    
    return grn_adata_filtered



def create_regulon_activity_adata(data, grn_adata, grn_adata_):
    """
    Convert processed regulon activity `data` into a new AnnData object aligned with GRN metadata.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame of regulon activity scores (e.g., UCell outputs). Columns are typically signature names.

    grn_adata : AnnData
        Full GRN AnnData object, used to inherit cell order, UMAP coordinates, and obs metadata.

    grn_adata_ : AnnData
        Filtered GRN AnnData object, used to select relevant source genes for columns.

    Returns:
    --------
    adata_regl : AnnData
        New AnnData object containing regulon activity with appropriate obs, var, and UMAP embeddings.
    """

    # Ensure column naming is consistent
    data.columns = [col.split("_")[0] + "_regulon" for col in data.columns]

    # Align naming convention for filtering
    expected_cols = [s + "_regulon" for s in grn_adata_.var["source"].unique() if s + "_regulon" in data.columns]
    data = data[expected_cols]

    # Create AnnData object
    adata_regl = ad.AnnData(
        X=data,
        obs=pd.DataFrame(index=data.index),
        var=pd.DataFrame(index=data.columns)
    )

    # Inherit obs and umap from grn_adata
    adata_regl = adata_regl[grn_adata.obs.index]
    adata_regl.obsm['X_umap'] = grn_adata.obsm['X_umap']
    adata_regl.obs = grn_adata.obs.copy()

    return adata_regl
