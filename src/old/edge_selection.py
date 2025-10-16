import numpy as np
from statsmodels.stats.nonparametric import rank_compare_2indep


def get_differential_edges(attribution_anndata, percentile = 10):
    genelist = []
    if len(np.unique(attribution_anndata.obs['leiden']))>1 :
        for cat in np.unique(attribution_anndata.obs['leiden']):
            statisi =rank_compare_2indep(x1=attribution_anndata.X[attribution_anndata.obs['leiden']==cat], x2= attribution_anndata.X[attribution_anndata.obs['leiden']!=cat])
            sig_and_high = np.where((statisi.pvalue<(0.01/(attribution_anndata.X.shape[1]*attribution_anndata.X.shape[1])))  & (statisi.prob1>= 0.9))
            genelist = genelist+ list(sig_and_high[0])

    else:
        # FALLBACk
        m = np.abs(attribution_anndata.X).mean(axis=0)
        # Get the indices of genes in the top 10%
        top_10_percent_indices = np.where(m > np.percentile(m, 100-percentile))[0]

        # Get the indices of genes in the bottom 10%
        bottom_10_percent_indices = np.where(m < np.percentile(m, percentile))[0]

        # Combine the two arrays of indices and sort them
        genelist = np.unique(np.sort(
            np.concatenate((top_10_percent_indices, bottom_10_percent_indices))
        ))
    return genelist

def get_percentile_edges(attribution_anndata, percentile = 10):
    # FALLBACk
    m = attribution_anndata.X.mean(axis=0)
    # Get the indices of genes in the top 10%
    top_10_percent_indices = np.where(m > np.percentile(m, 100-percentile))[0]

    # Get the indices of genes in the bottom 10%
    bottom_10_percent_indices = np.where(m < np.percentile(m, percentile))[0]

    # Combine the two arrays of indices and sort them
    genelist = np.unique(np.sort(
        np.concatenate((top_10_percent_indices, bottom_10_percent_indices))
    ))
    return genelist

def get_edges(attribution_anndata, use_differential=False, percentile = 10):
    if use_differential:
        return get_differential_edges(attribution_anndata, percentile=percentile)
    else:
        return get_percentile_edges(attribution_anndata, percentile=percentile)