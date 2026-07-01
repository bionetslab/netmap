"""Gene co-expression clustering via hierarchical linkage on the GRN attribution matrix."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from statsmodels.formula.api import quantreg
import anndata as ad

def get_hierarchical_clustering(adata, genes=None):
    """Perform hierarchical clustering of genes by pairwise correlation.

    Computes the Pearson correlation matrix across the gene axis of ``adata.X``,
    converts it to a distance matrix, and fits an average-linkage dendrogram.
    Returns the raw correlation/cophenet pairs for downstream cutoff selection
    and the linkage object for tree-cutting.

    Args:
        adata (anndata.AnnData): AnnData object containing gene expression (or
            attribution) data in layer ``X``. Rows are cells, columns are genes.
        genes (list or None): Subset of gene names (matching ``adata.var.index``)
            to use for clustering. Defaults to ``None``, which uses all genes.

    Returns:
        tuple: A two-element tuple ``(df, dist_linkage)`` where:

            - ``df`` (pd.DataFrame): DataFrame with columns ``'cophenet'``
              (cophenetic distances from the linkage) and ``'corr'`` (upper
              triangular Pearson correlations, diagonal excluded).
            - ``dist_linkage``: Linkage matrix from ``scipy.cluster.hierarchy.average``
              on the correlation-distance matrix.
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


    return df, dist_linkage



def add_clusters_to_adata(adata, dist_linkage, cutoff_distance, cluster_var='correlation_cluster', genes=None):
    """Cut the dendrogram and annotate ``adata.var`` with flat cluster assignments.

    Applies ``scipy.cluster.hierarchy.fcluster`` at ``cutoff_distance`` to the
    provided linkage matrix and merges the resulting integer cluster labels into
    ``adata.var`` under ``cluster_var``. If ``cutoff_distance`` is ``None``,
    a ``-1`` sentinel column is added instead and a warning is printed.

    Args:
        adata (anndata.AnnData): AnnData object whose ``.var`` will receive the
            cluster label column. Must be the same object (or compatible) as the
            one used to compute ``dist_linkage``.
        dist_linkage: Linkage matrix as returned by
            :func:`get_hierarchical_clustering` (or any
            ``scipy.cluster.hierarchy`` linkage function).
        cutoff_distance (float or None): Height at which to cut the dendrogram.
            Pass ``None`` to skip clustering and fill with ``-1``.
        cluster_var (str): Name of the column added to ``adata.var``. Defaults
            to ``'correlation_cluster'``.
        genes (list or None): Gene subset used when building ``dist_linkage``;
            restricts which rows of ``adata.var`` are matched. Defaults to
            ``None`` (all genes).

    Returns:
        anndata.AnnData: The input ``adata`` with ``cluster_var`` merged into
        ``.var`` (outer join, so genes absent from the linkage receive NaN).
    """

    if genes is not None:
        adata_sub = adata_sub[:, adata.var.index.isin(genes)]
    else:
        adata_sub = adata

    if cutoff_distance is not None:
        # Cut the dendrogram and get cluster IDs
        clusters = hierarchy.fcluster(dist_linkage, t=cutoff_distance, criterion='distance')

        dfclu = pd.DataFrame({'index':adata_sub.var.index, cluster_var : clusters})
        dfclu = dfclu.set_index('index')

        adata.var = adata.var.merge(dfclu, left_index=True, right_index=True, how='outer')

    else:
        print("Clustering was not performed due to an invalid cutoff.")
        adata.var['cluster_id'] = pd.Categorical([-1] * adata.n_vars)

    return adata


def compute_regression(df, cophenet_threshold, dist_linkage, correlation_threshold=0.6, quantile=0.1):
    """Fit a quantile regression to determine an automatic dendrogram cut distance.

    Filters the cophenetic/correlation scatter to rows below ``cophenet_threshold``,
    fits a quantile regression line at ``quantile``, and solves for the cophenet
    value at which the fitted line crosses ``correlation_threshold``. Calls
    :func:`plot_regression_and_dendrogram` for visual validation.

    Args:
        df (pd.DataFrame): DataFrame with columns ``'cophenet'`` and ``'corr'``
            as returned by :func:`get_hierarchical_clustering`.
        cophenet_threshold (float): Upper bound on cophenet distance used to
            restrict the regression fit to a linear region of the scatter.
        dist_linkage: Linkage matrix passed through to
            :func:`plot_regression_and_dendrogram` for dendrogram rendering.
        correlation_threshold (float): Pearson correlation value at which the
            automatic cut distance is resolved. Defaults to 0.6.
        quantile (float): Quantile for the regression fit (0–1). Defaults to
            0.1 (10th percentile).

    Returns:
        float or None: The computed cutoff distance, or ``None`` if the
        regression slope is zero (degenerate fit).
    """

    # Filter the data based on the tree_threshold for the regression fit
    df_filtered = df[df.cophenet < cophenet_threshold]

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

    return cutoff_distance



def plot_regression_and_dendrogram(df, model, cutoff_distance, correlation_threshold, dist_linkage):
    """Plot the hierarchical clustering dendrogram alongside the quantile regression fit.

    Produces a two-panel figure: the left panel shows the dendrogram with the
    cut distance marked; the right panel shows the cophenetic/correlation scatter
    with the fitted quantile regression line and the automatic cutoff lines.

    Args:
        df (pd.DataFrame): Filtered DataFrame with columns ``'cophenet'`` and
            ``'corr'`` used for the regression (already subset to the linear
            region by :func:`compute_regression`).
        model: Fitted ``statsmodels`` quantile regression result object with a
            ``predict`` method and a ``params`` attribute.
        cutoff_distance (float or None): Horizontal/vertical cutoff lines drawn
            on both panels. Skipped if ``None``.
        correlation_threshold (float): Horizontal dashed line drawn on the
            regression panel to mark the target correlation level.
        dist_linkage: Linkage matrix passed to
            ``scipy.cluster.hierarchy.dendrogram``.

    Returns:
        None: Displays the figure via ``plt.show()``.
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


def plot_scatter_plot(df):
    """Plot a scatter of cophenetic distance vs Pearson correlation for cutoff inspection.

    Renders a single-panel scatter plot of the full (unfiltered) cophenetic/
    correlation pairs to help identify the ``cophenet_threshold`` value where the
    linear relationship breaks down before calling :func:`compute_regression`.

    Args:
        df (pd.DataFrame): DataFrame with columns ``'cophenet'`` and ``'corr'``
            as returned by :func:`get_hierarchical_clustering`.

    Returns:
        None: Displays the figure via ``plt.show()``.
    """
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))


    # Right Plot: Quantile Regression
    ax2.scatter(df['cophenet'], df['corr'], alpha=0.7, label='Data Points')


    ax2.set_title('Cophenet Correlation Scatter plot')
    ax2.set_xlabel('Cophenet')
    ax2.set_ylabel('Correlation')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


