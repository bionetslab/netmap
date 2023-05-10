import pandas as pd
import numpy as np
import scipy as sc
import os.path as op
import os
from expceptions import ColumnValidationError


def save_node_list(dataframe:pd.DataFrame, filename:str, overwrite=True)-> None:

    """
    Saves a list of nodes to file. Requires the column 'node' to be present in the column names.
    More columns are possible. The function is deliberately opinionated on the column names.

    Arguments:
    --------------------------------
    dataframe: a pandas data frame containing the column node
    filename: A fully qualified path
    overwrite: If true, files will be overwritten.
    """
    
    if op.exists(filename) and not overwrite:
        raise FileExistsError('File exists: {}'.format(filename))
    
    if not 'node' in dataframe.columns:
        raise ColumnValidationError('Missing columns ["node"]')

    dataframe.to_csv(path = filename, sep='\t', index=False)


def save_egde_list(dataframe:pd.DataFrame, filename:str, overwrite=True) -> None:

    """
    Saves a list of edged to file. This function is deliberately opinionated on 
    the names of the columns, source_node, target_node, directed and edge_weight are required.
    More columns are possible.

    Arguments:
    ----------------------------
    dataframe: A Pandas data frame containing the columns: source_node, target_node, directed, edge_weight
    filename: A fully qualified path 
    overwrite: If true files will be verwritten.
    """

    if not all(elem in dataframe.columns for elem in ['source_node', 'target_node', 'directed', "edge_weight"]):
        raise ColumnValidationError('Missing columns ["source_node", "target_node", "directed", "edge_weight"]')
    
    if op.exists(filename) and not overwrite:
        raise FileExistsError('File exists: {}'.format(filename))
    
    dataframe.to_csv(path = filename, sep='\t', index=False)


