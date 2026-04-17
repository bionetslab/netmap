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
    """Calculates Jaccard similarity between a query set and a dict of reference sets."""
    results = {}
    q_set = set(query_set)
    for name, ref_set in reference_dict.items():
        intersection = len(q_set.intersection(ref_set))
        union = len(q_set.union(ref_set))
        results[name] = intersection / union if union > 0 else 0
    return results

def get_grn_targets(keep_edges, cluster, source_genes):
    """Extracts target genes for specific sources from the filtered edge object."""
    re = keep_edges['unique'][cluster]['edges']
    return list(re[re.source.isin(source_genes)].target)



def prepare_jaccard_analysis_df(grn_adata, all_signatures, keep_edges, marker_sets, cluster_mapper):
    """Processes all clusters and returns a unified DataFrame for plotting."""
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
    """Creates the faceted boxplot with refined aesthetics."""
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