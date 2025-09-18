import sys
sys.path.append('/data_nfs/og86asub/netmap/netmap-evaluation/')

import scanpy as sc
import time 
import os.path as op
import os
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import scipy.sparse as scs

import torch
import yaml


from netmap.src.utils.misc import write_config


from netmap.src.utils.data_utils import *
from netmap.src.utils.tf_utils import *
from netmap.src.utils.netmap_config import NetmapConfig

from netmap.src.model.train_model import create_model_zoo
from netmap.src.grn.inferrence import inferrence
from src.data_simulation.data_simulation_config import DataSimulationConfig
from netmap.src.masking.internal import *
from netmap.src.masking.external import *

def read_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config





def run_netmap(config, dataset_config):

    print('Version 2')
    start_total = time.monotonic()
    
    ## Load config and setup outputs
    os.makedirs(config.output_directory, exist_ok=True)
    sc.settings.figdir = config.output_directory
    config.write_yaml(yaml_file=op.join(config.output_directory, 'config.yaml'))

    ## load data
    adata = sc.read_h5ad(config.input_data)
    

    ## Get the data matrix from the CustumAnndata obeject

    gene_names = np.array(adata.var.index)
    model_start = time.monotonic()

    if config.layer == 'counts':
        data_tensor = adata.layers['counts']
    else:
        data_tensor = adata.X

    if scs.issparse(data_tensor):
        data_tensor = torch.tensor(data_tensor.todense(), dtype=torch.float32)
    else:
        data_tensor = torch.tensor(data_tensor, dtype=torch.float32)


    print(data_tensor.shape)

    model_zoo = create_model_zoo(data_tensor,  n_models=config.n_models, n_epochs=config.epochs, model_type=config.model)

    grn_adata = inferrence(model_zoo, data_tensor.cuda(), gene_names,  config, use_raw_attribution=False)
    grn_adata_raw = inferrence(model_zoo, data_tensor.cuda(), gene_names,  config, use_raw_attribution=True)
    if config.xai_method == 'GradientShap':
        grn_adata.layers['raw_attribution'] = grn_adata_raw.X
        grn_adata.layers['raw_attribution_quantile_count'] = grn_adata_raw.layers['quantile_count']

    grn_adata = add_neighbourhood_expression_mask(adata,grn_adata)

    adob = adata.obs.reset_index()
    grn_adata.obs['cell_id'] = np.array(adob['cell_id'])
    grn_adata.obs['grn'] = np.array(adob['grn'])

    
    model_elapsed = time.monotonic()-model_start
    grn_adata.write_h5ad(op.join(config.output_directory,config.adata_filename))

    time_elapsed_total = time.monotonic()-start_total


    res = {'time_elapsed_total': time_elapsed_total, 'time_elapsed_netmap': model_elapsed} 
    write_config(res, file=op.join(config.output_directory, 'results.yaml'))

