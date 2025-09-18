import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from statsmodels.formula.api import quantreg
import anndata as ad

def get_hierarchical_clustering(adata, tree_threshold, correlation_threshold=0.6, quantile=0.1, genes=None, cluster_var = 'target_cluster'):
    """
    Performs hierarchical clustering, determines an optimal cutoff based on a
    quantile regression, and adds cluster information to adata.var.

    Args:
        adata (ad.AnnData): The input AnnData object.
        tree_threshold (float): The maximum cophenet distance to include in the regression fit.
        correlation_threshold (float): The minimum correlation value to define clusters.
        quantile (float): The quantile for the regression model (e.g., 0.1 for 10th percentile).
    """

    adata_sub = adata.copy()

    if genes is not None:
        adata_sub = adata_sub[:, adata.var.index.isin(genes)]
    # Ensure input is a numpy array
    X = adata_sub.X.T.copy()

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X)

    # Calculate distance matrix for hierarchical clustering
    corr_dist = 1 - corr_matrix
    dist_linkage = hierarchy.average(corr_dist)

    # Prepare data for quantile regression
    flatten_upper_triangular_excluding_diagonal = lambda m: m[np.triu_indices_from(m, k=1)]
    df = pd.DataFrame({
        'cophenet': hierarchy.cophenet(dist_linkage),
        'corr': flatten_upper_triangular_excluding_diagonal(corr_matrix)
    })
    
    # Filter the data based on the tree_threshold for the regression fit
    df_filtered = df[df.cophenet < tree_threshold]
    
    # Fit the quantile regression model
    low_quantile_model = quantreg('corr ~ cophenet', df_filtered).fit(q=quantile)
    
    # Calculate the intersection to determine the automatic cutoff
    intercept = low_quantile_model.params['Intercept']
    slope = low_quantile_model.params['cophenet']
    
    cutoff_distance = None
    if slope == 0:
        print("Warning: The slope of the regression line is zero. Cannot compute automatic cutoff.")
    else:
        cutoff_distance = (correlation_threshold - intercept) / slope
        print(f"Automatically determined cluster cutoff distance: {cutoff_distance:.4f}")

    # Plot the regression and dendrogram for visual validation
    plot_regression_and_dendrogram(df_filtered, low_quantile_model, cutoff_distance, correlation_threshold, dist_linkage)
    
    if cutoff_distance is not None:
        # Cut the dendrogram and get cluster IDs
        clusters = hierarchy.fcluster(dist_linkage, t=cutoff_distance, criterion='distance')
        
        dfclu = pd.DataFrame({'index':adata_sub.var.index, cluster_var : clusters})
        dfclu = dfclu.set_index('index')
        print(adata.var.merge(dfclu, left_index=True, right_index=True, how='outer'))
        adata.var = adata.var.merge(dfclu, left_index=True, right_index=True, how='outer')


    else:
        print("Clustering was not performed due to an invalid cutoff.")
        adata.var['cluster_id'] = pd.Categorical([-1] * adata.n_vars)
    
    return adata

def plot_regression_and_dendrogram(df, model, cutoff_distance, correlation_threshold, dist_linkage):
    """
    Plots the data points, quantile regression line, automatic cutoff, and the dendrogram.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Left Plot: Dendrogram
    hierarchy.dendrogram(dist_linkage, color_threshold=cutoff_distance, ax=ax1)
    ax1.set_title('Hierarchical Clustering Dendrogram')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Distance')
    if cutoff_distance is not None:
        ax1.axhline(y=cutoff_distance, color='r', linestyle='--', label=f'Cutoff Distance: {cutoff_distance:.4f}')
        ax1.legend()

    # Right Plot: Quantile Regression
    ax2.scatter(df['cophenet'], df['corr'], alpha=0.7, label='Data Points')
    x_sorted = np.sort(df['cophenet'])
    y_predicted = model.predict({'cophenet': x_sorted})

    ax2.plot(x_sorted, y_predicted, color='red', linewidth=2, label='10th Percentile Quantile Regression Line')
    
    # Plot the automatic cutoff lines
    if cutoff_distance is not None:
        ax2.axhline(y=correlation_threshold, color='r', linestyle='--', label=f'Correlation Threshold: {correlation_threshold}')
        ax2.axvline(x=cutoff_distance, color='b', linestyle='--', label=f'Automatic Cutoff: {cutoff_distance:.4f}')

    ax2.set_title('Cophenet Correlation and Automatic Cutoff Determination')
    ax2.set_xlabel('Cophenet')
    ax2.set_ylabel('Correlation')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

