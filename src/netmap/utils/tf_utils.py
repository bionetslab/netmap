import pandas as pd
import anndata



def load_tf_names(path):
    """
    Load a transcription factor file with one single column of transcription
    factors.

    Args:
    path: Full path to the transcription factor file.
    """
    tfs_genes = pd.read_csv(path, names=["tfs"])
    # tfs_genes = pd.read_csv("../allTFs_mm.txt", names=["tfs"])
    
    tfs_genes = list(tfs_genes["tfs"])
    return tfs_genes


def filter_tf_names(tf_genes: list, adata: anndata.AnnData, tfs_only=True):
    """
    Filter the transcription factor list using an anndata object

    Args:
    tf_genes: list of transcription factors
    adata: anndata.AnnData object
    tfs_only: Boolean indicating whether to return transcription factors, or
                the full intersection of the adata.var.index and the tf_list
    """
    tf_genes = [g for g in tf_genes if g in list(adata.var_names)]
    tf_genes = set(tf_genes).intersection(set(adata.var_names))

    if tfs_only:
        tfs_indexes = [adata.var.index.get_loc(name) for name in tf_genes]
        print(f"number of  TFs: {len(tfs_indexes)}")
        if len(tfs_indexes) == 0:
            raise ValueError("No gene and TF overlap")
    else:
        print("Not usig TFs only aka GRN mode.")
        tfs_indexes = [i for i in range(len(adata.var_names))]
        tf_genes = adata.var_names
        print(f"number of  genes used as TFs: {len(tfs_indexes)}")
    return tfs_indexes, tf_genes


def get_tf_index(tf_genes, gene_names, tfs_only=True):
    """
    Returns the indices of transcription factors in a gene list.

    tf_genes: List of transcription factor names
    gene_names: List of gene names to search the tfs in.
    tfs_only: Whether to restrict to transcription factors or return the full
                list of indices and genes.
    """
    tf_genes = [g for g in tf_genes if g in list(gene_names)]
    tf_genes = set(tf_genes).intersection(set(gene_names))

    if tfs_only:
        tfs_indexes = [i for i in range(len(gene_names)) if gene_names[i] in tf_genes]
        tf_names = [i for i in gene_names if i in tf_genes]
        print(f"number of  TFs: {len(tfs_indexes)}")
        if len(tfs_indexes) == 0:
            raise ValueError("No gene and TF overlap")
    else:
        print("Not usig TFs only aka GRN mode.")
        tfs_indexes = [i for i in range(len(gene_names))]
        tf_names = gene_names
        print(f"number of  genes used as TFs: {len(tfs_indexes)}")
    return tfs_indexes, tf_names

