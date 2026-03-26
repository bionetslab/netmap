import scanpy as sc
import pandas as pd

def rank_regulon_groups_dotplot(grn_adata_filtered, adata_regl, original_cluster_column = 'leiden', new_cluster_column = 'leiden_remap',  n_genes=10, key="wilcoxon",  cmap='bwr', figsize=(25, 2), values_to_plot="scores", return_fig = True):
    """_summary_

    Function will throw and error if original cluster column and new cluster are not the same.


    Args:
        grn_adata_filtered (_type_): _description_
        adata_regl (_type_): _description_
        original_cluster_column (str, optional): _description_. Defaults to 'leiden'.
        new_cluster_column (str, optional): _description_. Defaults to 'leiden_remap'.


    Returns:
        _type_: _description_
    """


    cluster_mapping = grn_adata_filtered.obs[[original_cluster_column, new_cluster_column]].drop_duplicates()
    leiden_to_celltype = dict(zip(cluster_mapping[original_cluster_column], cluster_mapping[new_cluster_column]))
    colheaders = grn_adata_filtered.var.columns[grn_adata_filtered.var.columns.str.contains('nonzero')]

    colheaders = list(colheaders)
    if 'count_nonzero' in colheaders:
        colheaders.remove('count_nonzero')
    if 'count_nonzero_norm' in colheaders:
        colheaders.remove('count_nonzero_norm')

    adata_regl.var['regulon_name'] = [x.replace('_UCell', '') for x in adata_regl.var.index]
    adata_regl.var['source'] = [x.split('_')[-1] for x in adata_regl.var['regulon_name']]

    fractions = {}
    count = 0
    for ri in range(len(adata_regl.var.regulon_name)):
        regulon = adata_regl.var.regulon_name[ri]
        sou = adata_regl.var.source[ri]
        count = count+1
        fractions[f'{regulon}'] =   grn_adata_filtered.var[grn_adata_filtered.var.source == sou][colheaders].sum()
        
    fractions = pd.DataFrame(fractions)
    fractions.index = [x.replace('_nonzero', '') for x in fractions.index]

    # return fig needs to be true: get plot, modify sizes, then plot or return
    pp = sc.pl.rank_genes_groups_dotplot(adata_regl, n_genes=n_genes, key=key, groupby=new_cluster_column, cmap=cmap, figsize=figsize, values_to_plot=values_to_plot, return_fig = True)
    fractions = fractions.reindex(list(pp.dot_size_df.index))

    pp.dot_size_df = fractions.loc[:, pp.dot_color_df.columns]
    pp.dot_size_df = pp.dot_size_df/(pp.dot_size_df.max())
    pp.dot_size_df = pp.dot_size_df.fillna(0)
    
    if return_fig:
        return pp
    else:
        pp.show()



import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def get_grn_from_regulon(regulon_df, full_name, top_n=20):
    """
    Parses 'nk_cells_NKG7' into cluster='nk_cells' and source='NKG7',
    then builds the DiGraph.
    """
    # Split from the right to handle cluster names with underscores (e.g., cd8+_tcells)
    cluster, source = full_name.rsplit('_', 1)
    
    subset = regulon_df[
        (regulon_df.set_type == 'unique') & 
        (regulon_df.cluster == cluster) & 
        (regulon_df.source == source)
    ].nlargest(top_n, 'sum_of_edge')
    
    return nx.from_pandas_edgelist(subset, 'source', 'target', create_using=nx.DiGraph())

def draw_inset_graph(parent_ax, G, orientation='x'):
    """Handles the geometry of the marginal GRN plots."""
    bbox = (0.0, -0.45, 1.0, 0.3) if orientation == 'x' else (-0.45, 0.0, 0.3, 1.0)
    ax_ins = inset_axes(parent_ax, width="100%", height="100%", loc='center',
                        bbox_to_anchor=bbox, bbox_transform=parent_ax.transAxes)
    
    pos = nx.spring_layout(G, k=1.5, seed=42)
    nx.draw_networkx(G, pos, ax=ax_ins, node_size=200, node_color='#a8dadc', 
                     edge_color='#457b9d', alpha=0.7, font_size=7, font_weight='bold')
    ax_ins.axis('off')

def plot_regulon_comparison(adata, regulon_table, regulons, cluster_key='leiden_remap'):
    """
    The 'One-Liner' function.
    Pass it the adata, the big regulon table, and the two strings.
    """
    # 1. Prepare Scatter Data
    df = pd.DataFrame(adata[:, regulons].X.copy(), columns=regulons)
    df['group'] = adata.obs[cluster_key].values

    # 2. Build Graphs automatically from the names
    G_x = get_grn_from_regulon(regulon_table, regulons[0])
    G_y = get_grn_from_regulon(regulon_table, regulons[1])

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    sns.scatterplot(data=df, x=regulons[0], y=regulons[1], hue='group', ax=ax, s=20, alpha=0.5)
    
    draw_inset_graph(ax, G_x, 'x')
    draw_inset_graph(ax, G_y, 'y')
    
    # Clean up aesthetics
    sns.despine(ax=ax)
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return fig, ax


def compute_bcrank_metrics(bcrank_df, top_n=100):
    """
    Processes bcrank DataFrame and calculates entropy metrics.
    """
    # 1. Preprocessing
    df = bcrank_df.head(top_n).copy()
    if 'celltype' not in df.columns:
        df[['celltype', 'gene']] = df['names'].str.rsplit('_', n=1, expand=True)
    
    ranked_terms = df['celltype'].tolist()

    # 2. Entropy and Discovery Calculation
    raw_h = []
    discoveries = []
    seen = set()
    counts = Counter()
    
    for i, term in enumerate(ranked_terms, 1):
        if term not in seen:
            discoveries.append((i, term)) # Store (Rank, Term Name)
            seen.add(term)
        
        counts[term] += 1
        probs = [count / i for count in counts.values()]
        
        # Shannon Entropy calculation
        h = -sum(p * np.log2(p) for p in probs if p > 0)
        raw_h.append(h)
    
    # Store results in a dictionary for easy passing to plotting
    metrics = {
        'ranks': list(range(1, len(ranked_terms) + 1)),
        'raw_h': raw_h,
        'discoveries': discoveries,
        'top_n': top_n
    }
    
    return metrics

def plot_bcrank_entropy(metrics, title='Cell-Type Discovery Entropy', figsize=(12, 7), label_limit=12):
    """
    Visualizes the entropy metrics generated by compute_bcrank_metrics.
    """
    ranks = metrics['ranks']
    raw_h = metrics['raw_h']
    discoveries = metrics['discoveries']
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Plotting the curve
    ax.plot(ranks, raw_h, linewidth=2.5, color='#2c7bb6', label='Raw Entropy (H)', zorder=2)
    
    # Styling
    ax.set_xlabel('Gene Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shannon Entropy (bits)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)

    # Annotations
    for i, (rank, term) in enumerate(discoveries[:label_limit]):
        y_pos = raw_h[rank-1]
        ax.scatter(rank, y_pos, color='firebrick', s=50, edgecolors='white', zorder=5)
        
        # Vertical stagger logic
        v_offset = 30 + (i % 3 * 15)
        
        ax.annotate(
            term, 
            xy=(rank, y_pos), 
            xytext=(0, v_offset),
            textcoords='offset points',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8, connectionstyle="arc3,rad=0.1"),
            ha='center', fontsize=9, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8, lw=0.5)
        )

    if raw_h:
        ax.set_ylim(0, max(raw_h) * 1.4)
        
    ax.set_title(f"{title} (Top {metrics['top_n']} Genes)", fontsize=14, pad=15, fontweight='bold')
    plt.show()

def bcrank_entropy(bcrank_df, top_n=100, label_limit=12, return_metrics=False, **plot_kwargs):
    """
    Wrapper to compute and plot entropy discovery in one call.
    """
    # Run calculation
    metrics = compute_bcrank_metrics(bcrank_df, top_n=top_n)
    
    # Run plotting
    plot_bcrank_entropy(metrics, label_limit=label_limit, **plot_kwargs)
    
    if return_metrics:
        return metrics
