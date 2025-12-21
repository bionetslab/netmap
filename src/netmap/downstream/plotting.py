import scanpy as sc
from sklearn.metrics.cluster import contingency_matrix
import pandas as pd
import seaborn as sns
import numpy as np
import os.path as op

def compute_contingency(adata, grn_adata, config, col_adata = 'spectral', col_grn_adata = 'spectral'):
    cm = contingency_matrix(adata.obs[col_adata], grn_adata.obs[col_grn_adata])
    cm = pd.DataFrame(cm)
    cm.columns = [f'grn_{i}' for i in np.unique(grn_adata.obs[col_grn_adata])]
    cm.index = [f'orig_{i}' for i in np.unique(adata.obs[col_adata])]
    cm.to_csv(op.join(config['results']['output_directory'], 'contingency.tsv'), sep = '\t')
    plt = sns.heatmap(cm, annot=True,  fmt='g')
    fig = plt.get_figure()
    fig.savefig(op.join(config['results']['output_directory'], 'contingency_map.pdf'))
    return cm


def plot_umaps(adata, grn_adata, config):
    ## overlay clustering 
    grn_adata.obs.index = adata.obs.index
    adata.obs['grn_on_gex'] = grn_adata.obs['spectral']
    sc.pl.umap(grn_adata, color=['leiden', 'spectral'], save='_grn.pdf', title='GRN')
    sc.pl.umap(adata, color=['leiden', 'spectral'], save='_gex.pdf', title= 'GEX')
    sc.pl.umap(adata, color=[ 'grn_on_gex'], save = '_grn_on_gex.pdf')

def plot_differential_expression(grn_adata, column = 'spectral_sub_remap', suffix = 'grn.pdf'):
    #sc.settings.figdir = config['results']['output_directory']
    grn_adata.var_names_make_unique()
    grn_adata.obs[column] = grn_adata.obs[column].astype('str').astype("category")
    sc.tl.rank_genes_groups(grn_adata, column,use_raw=False)
    sc.pl.rank_genes_groups_dotplot(grn_adata, n_genes=5,use_raw=False , save=suffix)
