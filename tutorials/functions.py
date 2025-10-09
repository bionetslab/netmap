import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from multiprocessing import Pool

def compute_single_chi2(args):
    """
    Helper function to compute chi-squared for a single pair of genes
    from a pre-computed contingency table.
    """
    contingency_table, genea_index, geneb_index = args
    
    # Chi-squared test and return p-value and statistic
    try:
        cont = chi2_contingency(contingency_table)
        return genea_index, geneb_index, cont.pvalue, cont.statistic
    except ValueError:
        return genea_index, geneb_index, np.nan, np.nan

def compute_stats_mat_parallel(df_binary):
    """
    Computes the chi-squared p-values and test statistics in parallel
    by first computing all crosstabs before parallelization.
    """
    loop_end = df_binary.shape[0]
    gene_names = df_binary.index
    
    pvalues = np.zeros((loop_end, loop_end))
    teststat = np.zeros((loop_end, loop_end))

    print("compute oontinencies")
    tasks = []
    for genea in range(loop_end):
        for geneb in range(genea + 1, loop_end):
            gene_a_data = df_binary.iloc[genea]
            gene_b_data = df_binary.iloc[geneb]
            
            # Create the contingency table using pandas.crosstab
            contingency_table = pd.crosstab(gene_a_data, gene_b_data)
            
            # Add the pre-computed table and indices to the task list
            tasks.append((contingency_table, genea, geneb))

    print('Compute tests')
    with Pool() as pool:
        results = pool.map(compute_single_chi2, tasks)
    
    # Step 3: Populate the result matrices
    for genea, geneb, pvalue, statistic in results:
        pvalues[genea, geneb] = pvalue
        teststat[genea, geneb] = statistic
        
    # Symmetrize the matrices
    pvalues = pvalues + pvalues.T
    teststat = teststat + teststat.T

    np.fill_diagonal(pvalues, 1)
    np.fill_diagonal(teststat, 0)

    return pvalues, teststat, gene_names


import pandas as pd
import numpy as np

def binarize_and_compute_contingency(df_counts):
    """
    Binarizes raw gene count data and computes pairwise contingency tables efficiently.

    Args:
        df_counts (pd.DataFrame): A DataFrame with genes as rows and samples as columns.

    Returns:
        dict: A dictionary containing four DataFrames for the contingency table values:
              'a' (both genes expressed), 'b' (gene A expressed, gene B not),
              'c' (gene A not, gene B expressed), and 'd' (neither expressed).
    """

    # Step 1: Binarize the gene expression data
    # Convert counts to 1 (expressed) or 0 (not expressed)
    df_binary = (df_counts > 0).astype(int)

    # Step 2: Compute pairwise contingency using matrix multiplication
    # 'a': Both genes are expressed (count > 0 in the same sample)
    a_matrix = df_binary.dot(df_binary.T)

    # 'd': Neither gene is expressed (count == 0 in the same sample)
    # We invert the binary matrix to find where both are 0
    df_binary_inverted = 1 - df_binary
    d_matrix = df_binary_inverted.dot(df_binary_inverted.T)

    # Calculate marginal sums for 'b' and 'c'
    # Row-wise sum gives total number of samples where each gene is expressed
    expressed_counts = df_binary.sum(axis=1)

    # 'b': Gene A is expressed, Gene B is not
    # This is calculated as (total samples where Gene A is expressed) - (samples where both are expressed)
    b_matrix = expressed_counts.to_frame().values - a_matrix

    # 'c': Gene B is expressed, Gene A is not
    # This is calculated as (total samples where Gene B is expressed) - (samples where both are expressed)
    # We use a transpose here to align the dimensions correctly
    c_matrix = expressed_counts.to_frame().T.values - a_matrix

    # Correcting for floating point inaccuracies by rounding to nearest integer
    b_matrix = b_matrix.round().astype(int)
    c_matrix = c_matrix.round().astype(int)

    return {
        'a': pd.DataFrame(a_matrix, index=df_counts.index, columns=df_counts.index),
        'b': pd.DataFrame(b_matrix, index=df_counts.index, columns=df_counts.index),
        'c': pd.DataFrame(c_matrix, index=df_counts.index, columns=df_counts.index),
        'd': pd.DataFrame(d_matrix, index=df_counts.index, columns=df_counts.index)
    }




import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from multiprocessing import Pool

def compute_single_chi2(args):
    """
    Helper function to compute chi-squared for a single pair of genes.
    This function is designed to be used with multiprocessing.Pool.
    """
    contingency_tables, genea, geneb = args
    genenames = contingency_tables['a'].columns
    contingency_dict = {
        'expressed': {'expressed': contingency_tables['a'].loc[genenames[genea], genenames[geneb]],
                      'not_expressed': contingency_tables['b'].loc[genenames[genea], genenames[geneb]]},
        'not_expressed': {'expressed': contingency_tables['c'].loc[genenames[genea], genenames[geneb]],
                          'not_expressed': contingency_tables['d'].loc[genenames[genea], genenames[geneb]]}
    }
    contingency_df2 = pd.DataFrame(contingency_dict)
    try:
        cont = chi2_contingency(contingency_df2)
        return genea, geneb, cont.pvalue, cont.statistic
    except ValueError:
        return genea, geneb, np.nan, np.nan

def compute_stats_mat_parallel(contingency_tables):
    """
    Computes the chi-squared p-values and test statistics in parallel
    using the multiprocessing library.
    """
    loop_end = contingency_tables['a'].shape[1]
    pvalues = np.zeros((loop_end, loop_end))
    teststat = np.zeros((loop_end, loop_end))
    
    tasks = [(contingency_tables, genea, geneb) for genea in range(loop_end) for geneb in range(genea + 1, loop_end)]
    
    with Pool() as pool:
        results = pool.map(compute_single_chi2, tasks)
    
    for genea, geneb, pvalue, statistic in results:
        pvalues[genea, geneb] = pvalue
        teststat[genea, geneb] = statistic
        
    pvalues = pvalues + pvalues.T
    teststat = teststat + teststat.T


    np.fill_diagonal(pvalues, 1)
    np.fill_diagonal(teststat, 0)


    return pvalues, teststat, gene_names



