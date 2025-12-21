import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import scipy.sparse as scs


def random_score_distributions(grn_adata, percentile = 95, column_of_interest = 'spectral', rel_abs = False):
    """
    Create random score distributions across clusters via subsampling
    Compare the score of the edges within one cluster and select only
    those which fill above the p percentile of the randomized scores.
    Return the indices of those edges.
    
    """
    if rel_abs:
        grn_adata.X = np.abs(grn_adata.X)
    relevant_indices = {}
    
    for c in grn_adata.obs[column_of_interest].unique():
        background_sum_distribution = []
        for i in range(1000):
            samples = len(np.where(grn_adata.obs[column_of_interest] == c)[0])
            rs = np.random.choice( range(grn_adata.obs.shape[0]),size=samples, replace=False)
            sum_of_lrps = grn_adata.X[rs, :].sum(axis = 0)
            background_sum_distribution.append(sum_of_lrps)
        background_sum_distribution = np.stack(background_sum_distribution)
        background_sum_distribution = np.array(background_sum_distribution)
        percentiles = np.percentile(background_sum_distribution, axis = 0, q = percentile)
    
        sum_of_interest = grn_adata.X[grn_adata.obs[column_of_interest] == c, :].sum(axis = 0)
        #sum_of_interest = np.array(sum_of_interest).reshape(sum_of_interest.shape[1])
        relevant_indices[c] = np.where((sum_of_interest-percentiles)>0)[0]
    return relevant_indices



def unify_group_labelling(adata, grn_adata, col_adata, col_grn_adata, return_mapping=False):
    """
    Adjust group labelling such that grn_adata has the same group label than the 
    corresponding column in adata based on the grn column.

    Returns the data objects and a score of the matching as the cost of the matching
    divided by the number of cells.
    
    """

    cm = contingency_matrix(adata.obs[col_adata], grn_adata.obs[col_grn_adata])
    row_ind, col_ind = linear_sum_assignment(cm, maximize = True)
    
    names_ad = np.unique(adata.obs[col_adata])
    names_grn = np.unique(grn_adata.obs[col_grn_adata])
    mapping = {}
    reverse_mapping = {}
    for i in range(len(row_ind)):
        if ((isinstance(names_ad[0], str)) & (isinstance(names_grn[0], str))):
            reverse_mapping[names_grn[col_ind[i]]] = names_ad[row_ind[i]]
        elif (isinstance(names_ad[0], str)):
            reverse_mapping[col_ind[i]] = names_ad[row_ind[i]]
        elif (isinstance(names_grn[0], str)):
            reverse_mapping[names_grn[col_ind[i]]] = row_ind[i]
        else:
            reverse_mapping[col_ind[i]] = row_ind[i]
    print(reverse_mapping)
    print(mapping)
    col_grn_adata_remapped = col_grn_adata + '_remap'
    if isinstance(np.unique(grn_adata.obs[col_grn_adata])[0], str):
        grn_adata.obs[col_grn_adata_remapped] = [reverse_mapping[a] for a in grn_adata.obs[col_grn_adata]]
    else:
        grn_adata.obs[col_grn_adata_remapped] = [reverse_mapping[int(a)] for a in grn_adata.obs[col_grn_adata]]

    grn_adata.obs[col_grn_adata_remapped] = pd.Categorical(grn_adata.obs[col_grn_adata_remapped])

    score = cm[row_ind, col_ind].sum()/adata.obs.shape[0]
    if return_mapping:
        return adata, grn_adata, score, reverse_mapping
    else:
        return adata, grn_adata, score


def create_cluster_subset(grn_adata, grn_adata_sub, rel, cluster=0, obs = 'spectral', add_obs = ['spectral_remap']):
    ## subset only edges relevant in this cluster from original object
    sub1 = grn_adata[:, rel[int(cluster)]]
    #add the metadata from the remapped subset
    sub1.obs[[obs]+add_obs] = grn_adata_sub.obs[[obs]+add_obs] 
    # subset the correct group
    sub1 = sub1[sub1.obs[obs]==cluster]
    return sub1


def get_edges(sub1):
    sub1.var[['source','target']] =[x.split('_') for x in sub1.var.index.tolist() ]
    edges = scs.find(sub1.X)
    barcode = [sub1.obs.index[i] for i in edges[0]]
    source = [sub1.var.source[i] for i in edges[1]]
    target = [sub1.var.target[i] for i in edges[1]]
    edge = [sub1.var.index[i] for i in edges[1]]
    edgedf = pd.DataFrame({'barcode': barcode, 'edge': edge, 'source': source, 'target': target,  'value': edges[2]})
    
    summary = edgedf.groupby('edge').median('value')
    summary['mean'] = edgedf.groupby('edge').mean('value')
    summary = summary.rename(columns= {'value': 'median'})
    summary = summary.sort_values('median', ascending=False)
    return summary

def create_all_summaries(grn_adata, grn_adata_sub, rel, cluster_col = 'spectral'):
    summaries = []
    for c in np.unique(grn_adata_sub.obs[cluster_col]):
        sub1 = create_cluster_subset(grn_adata, grn_adata_sub, rel, cluster = c)
        summary = get_edges(sub1)
        summary[cluster_col] = c
        summaries.append(summary)
    summaries = pd.concat(summaries)
    return summaries