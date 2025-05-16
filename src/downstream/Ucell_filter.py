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

def Ucell_filter_(adata, grn_adata, cluster, top_per_source, min_regulon_size, ncores=100):
    

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

    print(f"N sources: {len(list(df_gene['source'].unique()))}")

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
    print(f"length Keep_this_edges: {len(list(set(Keep_this_edges)))}")


    print("\n")
    print(f"df_gene shape: {df_gene.shape}")
    all_edges = df_gene[df_gene.index.isin(Keep_this_edges)]
    print(f"all_edges shape: {all_edges.shape}")
    all_edges['edge'] = all_edges.index
    print(all_edges["source"] .value_counts())
    print(f"all_edges shape: {all_edges.shape}")
    print(all_edges.head(4))
    print("\n")
    
    print("In R")
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
    print(data.shape)
    
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
