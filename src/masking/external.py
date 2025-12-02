import pandas as pd
import numpy as np

import pandas as pd
from gtfparse import read_gtf
from sklearn.cluster import DBSCAN
from collections import defaultdict

def _create_edge_mask_from_GRN(grn_df, gene_list, name_grn='external_grn'):
    """
    Create flat vector mask for TF target interactions based on literature GRN and 
    gene list.
    
    Args:
        grn_df (pd.DataFrame): A DataFrame representing the GRN with columns 'source',
                               'target', and 'weight'.
        gene_list (list): A list of gene names to be included in the matrix. (should be the gene in the study)

    Returns:
        flat_vector: numpy.ndarray: numeric mask containing the value of the GRN
        names: numpy.ndarray: edge name vector (GeneA_GeneB)

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
    all_sources = np.unique(grnad.var.source)
    all_targets = np.unique(grnad.var.target)
    all_genes = np.unique(np.concatenate([all_sources, all_targets]))
    return all_genes


def add_external_grn(grn_ad, external_grn, name_grn = 'external_grn'):
    
    """
    Adds three columns to a anndate GRN object. 
    is_target
    is_source
    is_egde
    
    """

    all_my_genes = _get_all_genes_in_grn_object(grn_ad)
    edge_mask = _create_edge_mask_from_GRN(external_grn, all_my_genes, name_grn = name_grn)
    grn_ad.var = grn_ad.var.merge(edge_mask, left_index=True, right_index=True)
    grn_ad.var[f'is_target_{name_grn}'] = grn_ad.var.target.isin(external_grn.target)
    grn_ad.var[f'is_source_{name_grn}'] = grn_ad.var.source.isin(external_grn.source)
    return grn_ad



def get_genome_annotation_from_gtf(gtf_df):
    genes = gtf_df.filter(feature="gene")
    genes = pd.DataFrame(genes)
    genes.columns = gtf_df.columns
    genes = genes.loc[:, ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand',
            'frame', 'gene_id', 'gene_version', 'gene_name', 'gene_source',
            'gene_biotype']]
    genes = genes[genes["gene_name"]!='']
    genes['chr'] = 'chr'+genes['seqname']
    return genes


def preprocess_bed_file(bed_file, gtf_df):
    ## ALL cis regulatory motifs
    crm_df = pd.read_csv(bed_file, sep="\t", header=None)
    crm_df.columns = ['chr', 'start', 'end', 'TF_list','TF_number', 'strand', 'number1', 'number2', 'large_number']
    crm_by_chr = {chr_: df for chr_, df in crm_df.groupby("chr")}
    crm_df['TF_list_list'] = crm_df['TF_list'].str.split(",")
    return crm_df

        

def get_regulators(crm_df, genes, window):
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