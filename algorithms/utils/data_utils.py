import pandas as pd
import numpy as np
import scipy as sc
import scanpy as sp
import anndata as ad

def generate_random_pandas_data_frame(n, m):
    """
    Generates a random data frame of dimensions nxm with column names
    """
    data = np.random.random((n,m))
    data = pd.DataFrame(data)
    data.colums = ['column_'+str(i) for i in range(data.shape[1])]
    return data


        

def create_anndata_from_prefixes(data_directory:str, prefix:list) -> ad.AnnData:
    """
    This Basic Initializer reads one or more filed in 10X genomic mtx format
    and concatenates them into one large anndata object.

    --------------------------------------
    directory: location of the files
    prefix: prefix of the adata objects to be read
    """

    adata_list = []
    for pf in prefix:
        # Use scanpy's inbuilt read function
        data = sp.read_10x_mtx(data_directory, prefix=pf)
        # add a data entry for each data set
        data.obs['dataset'] = pf
        adata_list.append(data)

    adata = ad.concat(adata_list)    
    # set the data object 
    return adata

