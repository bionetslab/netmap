"""Jaccard-based comparison of GRN-derived regulon targets against cell-type marker sets."""

# --- Data Manipulation & Bioinformatics ---
import numpy as np
import pandas as pd
import scanpy as sc  # Required for sc.get.rank_genes_groups_df

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Standard Library (Math & Logic) ---
from collections import Counter
import math  # Optional, but np.log2 is used in the functions

def calculate_jaccard_scores(query_set, reference_dict):
    """Calculate Jaccard similarity between a query set and a dict of reference sets.

    Args:
        query_set (iterable): Gene names representing the query gene set.
        reference_dict (dict): Mapping of reference set names to sets (or iterables)
            of gene names to compare against the query.

    Returns:
        dict: Mapping of each key from ``reference_dict`` to its Jaccard similarity
        score with ``query_set``. Scores are floats in [0, 1]. A pair with an empty
        union receives a score of 0.
    """
    results = {}
    q_set = set(query_set)
    for name, ref_set in reference_dict.items():
        intersection = len(q_set.intersection(ref_set))
        union = len(q_set.union(ref_set))
        results[name] = intersection / union if union > 0 else 0
    return results

def get_grn_targets(keep_edges, cluster, source_genes):
    """Extract target genes for specific sources from the filtered edge object.

    Args:
        keep_edges (dict): Nested dict returned by ``select_top_edges()``, structured as
            ``{'unique': {cluster: {'edges': pd.DataFrame, 'summary': ...}, ...}, 'all': ...}``.
            The ``edges`` DataFrame must contain ``source`` and ``target`` columns.
        cluster: The cluster identifier used to look up the relevant edge DataFrame
            within ``keep_edges['unique']``.
        source_genes (iterable): Source gene names whose target genes should be retrieved.

    Returns:
        list: Target gene names (strings) regulated by any of the genes in
        ``source_genes`` within the specified cluster.
    """
    re = keep_edges['unique'][cluster]['edges']
    return list(re[re.source.isin(source_genes)].target)



def prepare_jaccard_analysis_df(grn_adata, all_signatures, keep_edges, marker_sets, cluster_mapper):
    """Process all clusters and return a unified DataFrame for plotting.

    For each Leiden cluster in ``grn_adata``, retrieves the top differentially
    expressed source genes from ``all_signatures``, fetches their GRN target
    genes via ``keep_edges``, and computes Jaccard similarity of both sets
    against every cell-type marker set in ``marker_sets``. Clusters that raise
    an exception are skipped with a printed warning.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData object (obs = cells, var = edges)
            whose ``obs.leiden_remap`` column lists the cluster label for every cell.
        all_signatures (anndata.AnnData): AnnData object containing a Wilcoxon
            rank_genes_groups result stored under the key ``'wilcoxon'`` (produced by
            ``sc.tl.rank_genes_groups``). Gene names are expected in the format
            ``<celltype>_<gene>`` so they can be split on the last underscore.
        keep_edges (dict): Nested dict returned by ``select_top_edges()``, structured as
            ``{'unique': {cluster: {'edges': pd.DataFrame, ...}}, 'all': ...}``.
        marker_sets (dict): Mapping of ``{celltype: set(genes)}`` mapping cell-type labels
            to their canonical marker gene sets used as Jaccard references.
        cluster_mapper (dict or pd.Series): Mapping from raw cluster identifiers to
            human-readable labels. Currently passed through but not applied internally;
            reserved for downstream label remapping.

    Returns:
        pd.DataFrame: One row per (cluster, marker-set, gene-set-type) combination with
        the following columns:

            - ``celltype`` (str): Key from ``marker_sets`` used as the reference.
            - ``jaccard`` (float): Jaccard similarity score in [0, 1].
            - ``ct`` (str): Cluster identifier from ``grn_adata.obs.leiden_remap``.
            - ``type`` (str): Either ``'Source'`` or ``'Target'``, indicating which
              gene set was compared.
    """
    all_results = []

    for ct in grn_adata.obs.leiden_remap.unique():
        try:
            # 1. Get Top Sources (Genes)
            bcrank = sc.get.rank_genes_groups_df(all_signatures, group=ct, key='wilcoxon')
            bcrank[['celltype', 'gene']] = bcrank['names'].str.rsplit('_', n=1, expand=True)
            top_sources = bcrank.head(20)['gene'].tolist()

            # 2. Get Targets of those Sources
            top_targets = get_grn_targets(keep_edges, ct, top_sources)

            # 3. Calculate Jaccard for both Source and Target sets
            src_jaccard = calculate_jaccard_scores(top_sources, marker_sets)
            tgt_jaccard = calculate_jaccard_scores(top_targets, marker_sets)

            # 4. Convert to DataFrames
            for label, scores in [('Source', src_jaccard), ('Target', tgt_jaccard)]:
                df = pd.DataFrame({'celltype': list(scores.keys()), 'jaccard': list(scores.values())})
                df['ct'] = ct
                df['type'] = label
                all_results.append(df)
        except Exception as e:
            print(f"Skipping cluster {ct} due to error: {e}")

    full_df = pd.concat(all_results)

    return full_df


def plot_jaccard_comparison(plot_df, title="Jaccard Similarity: On-Target vs Off-Target"):
    """Create a faceted boxplot comparing Jaccard scores across clusters and gene-set types.

    Renders a ``seaborn.catplot`` with one column per gene-set type (``'Source'``
    and ``'Target'``), x-axis showing cluster labels, y-axis showing Jaccard
    similarity, and hue encoding whether the cluster is mapped (``is_mapped``
    column). Tick labels are rotated for readability and the legend is
    repositioned to avoid overlap.

    Args:
        plot_df (pd.DataFrame): DataFrame as returned by
            :func:`prepare_jaccard_analysis_df`, expected to contain at minimum
            the columns ``ct``, ``jaccard``, ``is_mapped``, and ``type``.
        title (str): Figure-level super-title. Defaults to
            ``"Jaccard Similarity: On-Target vs Off-Target"``.

    Returns:
        seaborn.FacetGrid: The grid object containing the rendered figure.
        The figure can be saved with ``g.fig.savefig(...)`` or displayed with
        ``plt.show()``.
    """
    sns.set_style("ticks")

    g = sns.catplot(
        data=plot_df, x='ct', y='jaccard', hue='is_mapped',
        col='type', kind='box', palette='Set2',
        height=4, aspect=1.2, linewidth=1.2, fliersize=2, sharey=True
    )

    # Formatting
    g.set_xticklabels(rotation=45, ha='right')
    g.set_axis_labels("", "Jaccard Similarity", fontsize=12)
    g.set_titles("{col_name}", fontweight='bold', pad=10)

    sns.move_legend(g, loc="upper left", bbox_to_anchor=(0.1, 0.85), title="", frameon=True)

    for ax in g.axes.flat:
        ax.tick_params(axis='x', length=0)
        sns.despine(ax=ax)

    g.fig.suptitle(title, fontweight='bold', fontsize=14, y=1.05)
    return g
