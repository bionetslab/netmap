import scanpy as sc
import pandas as pd
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt



import networkx as nx

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def extract_tf_subgraph(filtered_df, tfs):
    
    g = ig.Graph.TupleList(filtered_df[['source', 'target', 'sum_of_edge']].itertuples(index=False), directed=False, weights=True)

    # 7. LAYOUT & PLOTTING
    layout = g.layout_kamada_kawai()

    if tfs is not None:
        g.vs["is_tf"] = [name in tfs for name in g.vs["name"]]
        colors = ["#ff7f0e" if is_tf else "#1f77b4" for is_tf in g.vs["is_tf"]]
    else:
        colors = ["#ff7f0e" for node in g.vs["is_tf"]]


    tf_indices = [v.index for v in g.vs if v["is_tf"]]
    tf_subgraph = g.subgraph(tf_indices)

    # 2. Get the largest connected component
    components = tf_subgraph.connected_components(mode='weak')
    giant_component = components.giant()

    # --- FIX: Ensure the weight attribute exists ---
    # If 'sum_of_edge' is missing, it might be stored simply as 'weight' 
    # because of how TupleList was initialized.
    attr_name = 'sum_of_edge' if 'sum_of_edge' in giant_component.es.attributes() else 'weight'

    try:
        edge_weights = giant_component.es[attr_name]
    except KeyError:
        # Fallback if no weights exist at all
        edge_weights = [1] * len(giant_component.es)

    # 3. Plotting
    layout = giant_component.layout_kamada_kawai()
    fig, ax = plt.subplots(figsize=(8, 8))

    ig.plot(
        giant_component,
        target=ax,
        layout=layout,
        vertex_size=25,
        vertex_color="#ff7f0e", 
        vertex_label=giant_component.vs["name"],
        vertex_label_size=12,
        vertex_label_dist=2.0,
        edge_alpha=1,

        edge_arrow_size=0.5
    )

    plt.title("Largest Connected TF Regulatory Module", fontsize=20)
    plt.show()