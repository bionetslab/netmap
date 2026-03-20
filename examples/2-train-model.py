
import anndata as ad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.pyplot import rc_context
import numpy as np
import sys
import os.path as op

import anndata
from netmap.downstream import final_downstream


import warnings

from netmap.utils.data_utils import *
from netmap.utils.tf_utils import *
from netmap.utils.netmap_config import NetmapConfig

from netmap.model.train_model import create_model_zoo
from netmap.grn.inferrence import inferrence, inferrence_model_wise
from netmap.masking.internal import *
from netmap.masking.external import *

from netmap.downstream.edge_selection import *
from netmap.downstream.clustering import *
from netmap.downstream.final_downstream import *

import scipy.sparse as scs
import torch



def train_model(adata, output_dir, model_name):

   
    gene_names = np.array(adata.var.index)
    data_tensor = adata.X # Log normalized, but not standardized data.


    if scs.issparse(data_tensor):
        data_tensor = torch.tensor(data_tensor.todense(), dtype=torch.float32)
    else:
        data_tensor = torch.tensor(data_tensor, dtype=torch.float32)

    model_zoo = create_model_zoo(data_tensor,  n_models=10, n_epochs=10000, model_type='NBAutoencoder', latent_dim= 8, dropout_rate=0.1, hidden_dim = [64] )

    grn_adata = inferrence(model_zoo, data_tensor.cuda(), gene_names, xai_method='GradientShap', background_type = 'zeros', backing_file=op.join(output_dir, f'{model_name}.parquet'), return_in_memory=False)
    
    #Save anndata obs to grn obs for reference
    grn_adata.obs = adata.obs
    grn_adata.write_h5ad( op.join(output_dir, f'{model_name}_grn.h5ad'))
    
    grn_adata.var.to_csv(op.join(output_dir, f'{model_name}_var.tsv'), header = '\t')
    # save the original obs
    adata.obs.to_csv(op.join(output_dir, f'{model_name}_obs.tsv'), header = '\t')



if __name__=='__main__':

    # define your output dir.
    output_dir = "netmap/case_studies/blood"
    os.makedirs(output_dir, exist_ok=True)

    # this is the folder and filename
    model_name = 'bd-rhap-rep1-X'
    model_output_dir = op.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    ## load your data
    adata = sc.read_h5ad('netmap/data/blood/reprocessed/bd-rhap-rep1.h5ad')
    # Use the correct layer!
    # Here we use the X layer. The data needs to be sc.pp.normalized(target = 10000) and sc.pp.log1() transformed
    # But not scaled (!sc.pp.scale())
    train_model(adata, output_dir=model_output_dir, model_name = model_name)

    