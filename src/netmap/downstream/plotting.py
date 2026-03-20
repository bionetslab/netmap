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

