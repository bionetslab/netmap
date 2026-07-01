"""External reference GRN integration and genome-proximity regulator annotation.

Functions in this module annotate GRN edges with:

- Overlap with a literature reference network (e.g. CollecTRI) via
  :func:`add_external_grn`.
- TF→gene relationships derived from genomic proximity (GTF + BED files)
  via :func:`get_regulators` and :func:`add_genome_information_to_anndata`.
"""

import pandas as pd
import numpy as np

import pandas as pd
from sklearn.cluster import DBSCAN
from collections import defaultdict

def _create_edge_mask_from_GRN(grn_df, gene_list, name_grn='external_grn'):
    """Create a flat vector mask for TF-target interactions based on a literature GRN and gene list.

    Builds a square gene-by-gene weight matrix from the reference GRN, flattens
    it into a vector, and returns a DataFrame indexed by edge names
    (``SourceGene_TargetGene``) suitable for merging into ``grn_adata.var``.

    Args:
        grn_df (pd.DataFrame): A DataFrame representing the GRN with columns
            ``source``, ``target``, and ``weight``.
        gene_list (list): A list of gene names to be included in the matrix.
            Should correspond to the genes present in the study.
        name_grn (str): Label used to name the output column
            ``edge_in_{name_grn}``. Defaults to ``'external_grn'``.

    Returns:
        pd.DataFrame: Single-column DataFrame indexed by edge key
        (``GeneA_GeneB``) with column ``edge_in_{name_grn}`` containing the
        numeric edge weight from the reference GRN (0 where no edge exists).
    """

    # Create a mapping from gene names to their matrix indices for efficient look-up.
    gene_to_index = {gene: i for i, gene in enumerate(gene_list)}
    num_genes = len(gene_list)

    matrix = np.zeros((num_genes, num_genes))

    # Iterate through each row of the GRN DataFrame and populate the matrix.
    for _, row in grn_df.iterrows():
        tf = row['source']
        target = row['target']
        weight = row['weight']

        # Check if both the TF and target are in our target gene list.
        if tf in gene_to_index and target in gene_to_index:
            tf_index = gene_to_index[tf]
            target_index = gene_to_index[target]

            # Populate the matrix with the corresponding weight.
            matrix[tf_index, target_index] = weight

    # Create edge names
    edge_names = []
    for gene_A in gene_list:
        for gene_B in gene_list:
            edge_names.append(f'{gene_A}_{gene_B}')

    edge_mask = matrix.flatten()
    edge_mask = pd.DataFrame({'edge_key': edge_names, f'edge_in_{name_grn}': edge_mask})
    edge_mask =edge_mask.set_index('edge_key')

    return edge_mask



def _get_all_genes_in_grn_object(grnad):
    """Returns the union of all source and target genes in the GRN AnnData.

    Args:
        grnad (anndata.AnnData): An AnnData object whose ``var`` DataFrame
            contains ``source`` and ``target`` columns identifying the two
            endpoints of each directed edge.

    Returns:
        np.ndarray: Sorted array of unique gene names appearing as either a
        source or a target across all edges in ``grnad.var``.
    """
    all_sources = np.unique(grnad.var.source)
    all_targets = np.unique(grnad.var.target)
    all_genes = np.unique(np.concatenate([all_sources, all_targets]))
    return all_genes


def add_external_grn(grn_ad, external_grn, name_grn='external_grn'):
    """Annotate GRN edges with overlap against an external reference network.

    For each edge in ``grn_ad``, adds a weight column from the reference GRN and
    boolean flags indicating whether the source or target gene appears in the
    reference.

    Args:
        grn_ad (anndata.AnnData): GRN AnnData whose ``.var`` contains ``source``
            and ``target`` columns.
        external_grn (pd.DataFrame): Reference GRN with ``source``, ``target``,
            and ``weight`` columns.
        name_grn (str): Suffix used to name the added ``.var`` columns:
            ``edge_in_{name_grn}``, ``is_target_{name_grn}``,
            ``is_source_{name_grn}``. Defaults to ``'external_grn'``.

    Returns:
        anndata.AnnData: ``grn_ad`` with three new columns in ``.var``.
    """
    all_my_genes = _get_all_genes_in_grn_object(grn_ad)
    edge_mask = _create_edge_mask_from_GRN(external_grn, all_my_genes, name_grn = name_grn)
    grn_ad.var = grn_ad.var.merge(edge_mask, left_index=True, right_index=True)
    grn_ad.var[f'is_target_{name_grn}'] = grn_ad.var.target.isin(external_grn.target)
    grn_ad.var[f'is_source_{name_grn}'] = grn_ad.var.source.isin(external_grn.source)
    return grn_ad



def get_genome_annotation_from_gtf(gtf_df):
    """Return gene-feature rows from a parsed GTF DataFrame.

    Filters the GTF to rows where ``feature == 'gene'``, selects a standard
    set of columns, removes entries with empty gene names, and prepends
    ``'chr'`` to the chromosome identifier.

    Args:
        gtf_df (pd.DataFrame): Genome annotation loaded from a GTF file,
            expected to contain standard GTF columns including ``seqname``,
            ``feature``, ``start``, ``end``, ``gene_id``, ``gene_name``, and
            ``gene_biotype``.

    Returns:
        pd.DataFrame: Filtered and reformatted DataFrame of gene features with
        an additional ``chr`` column containing the prefixed chromosome name.
    """
    genes = gtf_df.filter(feature="gene")
    genes = pd.DataFrame(genes)
    genes.columns = gtf_df.columns
    genes = genes.loc[:, ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand',
            'frame', 'gene_id', 'gene_version', 'gene_name', 'gene_source',
            'gene_biotype']]
    genes = genes[genes["gene_name"]!='']
    genes['chr'] = 'chr'+genes['seqname']
    return genes


def preprocess_bed_file(bed_file):
    """Read a BED file and parse the TF motif list column.

    Loads a tab-separated BED file describing cis-regulatory modules (CRMs)
    and splits the comma-separated TF list into a Python list for downstream
    use by ``get_regulators()``.

    Args:
        bed_file (str): Path to the BED file. Expected columns (0-indexed) are
            chr, start, end, TF_list, TF_number, strand, number1, number2,
            large_number.

    Returns:
        pd.DataFrame: DataFrame with the original BED columns plus an
        additional ``TF_list_list`` column containing each TF name as a
        Python list element.
    """
    ## ALL cis regulatory motifs
    crm_df = pd.read_csv(bed_file, sep="\t", header=None)
    crm_df.columns = ['chr', 'start', 'end', 'TF_list','TF_number', 'strand', 'number1', 'number2', 'large_number']
    crm_df['TF_list_list'] = crm_df['TF_list'].str.split(",")
    return crm_df



def get_regulators(crm_df, genes, window):
    """Identify TF regulators of each gene by proximity to the TSS.

    Searches a window of ``±window`` bp around each gene's TSS in the CRM
    (cis-regulatory motif) BED data and returns all TF→gene pairs found.

    Args:
        crm_df (pd.DataFrame): Preprocessed BED data from
            :func:`preprocess_bed_file`, with columns ``chr``, ``start``, ``end``,
            and ``TF_list_list``.
        genes (pd.DataFrame): Gene annotation table with columns ``chr``,
            ``start``, and ``gene_name`` (e.g. from
            :func:`get_genome_annotation_from_gtf`).
        window (int): Search window in base pairs upstream and downstream of the
            TSS.

    Returns:
        pd.DataFrame: Table with columns ``gene``, ``TFs``, ``nTFs``,
            ``edge`` (``TF_gene`` string), ``regulator`` (always ``True``).
    """
    gene_to_tfs = defaultdict(set)

    crm_by_chr = {chr_: df for chr_, df in crm_df.groupby("chr")}

    for idx, gene in genes.iterrows():
        chrom = gene["chr"]
        tss = gene["start"]
        gene_name = gene["gene_name"]

        if chrom not in crm_by_chr:
            continue

        crms = crm_by_chr[chrom]
        nearby_crms = crms[(crms["end"] >= tss - window) & (crms["start"] <= tss + window)]

        for _, crm in nearby_crms.iterrows():
            gene_to_tfs[gene_name].update(crm["TF_list_list"])



    results = pd.DataFrame([
        {"gene": gene, "TFs": sorted(list(tfs))}
        for gene, tfs in gene_to_tfs.items()
    ])

    results['nTFs'] = results['TFs'].apply(len)
    results = results.explode('TFs')
    results['edge'] = results['TFs'] + '_' + results['gene']

    results['regulator'] = True
    return results

def add_genome_information_to_anndata(grn_adata, tf_to_gene_df, window_name=''):
    """Merge TF→gene regulatory relationships into GRN AnnData variable metadata.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData object whose var index contains
            ``SourceGene_TargetGene`` edge identifiers.
        tf_to_gene_df (pd.DataFrame): Output of :func:`get_regulators` with at
            least ``edge`` and ``regulator`` columns.
        window_name (str): Suffix appended to the added boolean column name
            ``regulator_{window_name}``. Defaults to ``''``.

    Returns:
        anndata.AnnData: ``grn_adata`` with a new boolean column
            ``regulator_{window_name}`` in ``.var``.
    """
    grn_adata.var = grn_adata.var.reset_index().merge(tf_to_gene_df.loc[:, ['edge', 'regulator']], left_on='edge_key', right_on='edge', how='left').set_index('edge_key')
    grn_adata.var.regulator = grn_adata.var.regulator.apply(lambda x: False if pd.isna(x) else True)
    grn_adata.var = grn_adata.var.drop(columns = ['edge'])
    grn_adata.var = grn_adata.var.rename(columns= {'regulator': f'regulator_{window_name}'})
    return grn_adata
