"""Utilities for loading and indexing transcription factor lists against AnnData gene sets."""

import pandas as pd
import anndata



def load_tf_names(path):
    """Load a transcription factor file with one single column of transcription factors.

    Args:
        path: Full path to the transcription factor file.

    Returns:
        list of str: Transcription factor names.
    """
    tfs_genes = pd.read_csv(path, names=["tfs"])
    # tfs_genes = pd.read_csv("../allTFs_mm.txt", names=["tfs"])

    tfs_genes = list(tfs_genes["tfs"])
    return tfs_genes


def filter_tf_names(tf_genes: list, adata: anndata.AnnData, tfs_only=True):
    """Filter the transcription factor list using an AnnData object.

    Args:
        tf_genes (list of str): List of transcription factor names.
        adata (anndata.AnnData): AnnData object whose var_names define the
            gene universe.
        tfs_only (bool): If ``True``, return only transcription factors present
            in ``adata``; if ``False``, return the full set of genes from
            ``adata`` as pseudo-TFs.

    Returns:
        tuple: ``(tfs_indexes, tf_genes)`` where ``tfs_indexes`` is a list of
            integer positions in ``adata.var`` and ``tf_genes`` is the filtered
            set of TF names.
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
    """Return the indices and names of transcription factors within a gene list.

    Args:
        tf_genes (list of str): List of transcription factor names.
        gene_names (list of str): List of gene names to search for
            transcription factors.
        tfs_only (bool): If ``True``, restrict output to transcription factors
            present in ``gene_names``; if ``False``, return the full list of
            indices and genes.

    Returns:
        tuple: ``(tfs_indexes, tf_names)`` — integer indices and corresponding
            gene name list.
    """
    tf_genes = [g for g in tf_genes if g in list(gene_names)]
    tf_genes = set(tf_genes).intersection(set(gene_names))

    if tfs_only:
        tfs_indexes = [i for i in range(len(gene_names)) if gene_names[i] in tf_genes]
        tf_names = list(filter(tf_genes.__contains__, gene_names))
        print(f"number of  TFs: {len(tfs_indexes)}")
        if len(tfs_indexes) == 0:
            raise ValueError("No gene and TF overlap")
    else:
        print("Not usig TFs only aka GRN mode.")
        tfs_indexes = [i for i in range(len(gene_names))]
        tf_names = gene_names
        print(f"number of  genes used as TFs: {len(tfs_indexes)}")
    return tfs_indexes, tf_names
