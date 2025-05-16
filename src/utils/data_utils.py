import anndata
import os
import os.path as op

def attribution_to_anndata(attribution_list, varnames = None, obs = None)-> anndata.AnnData:

    """
    Transform attribution data frame into an anndata object

    Args:
    attribution_list: (sparse) Data frame of attribution values (one column per edge)

    returns: 
    Anndata object with attribution values in X.
    """
    adata = anndata.AnnData(attribution_list)
    adata.raw = adata
    if varnames is not None:
        adata.var.index = varnames
        adata.raw.var.index = varnames
    if obs is not None:
        adata.obs = obs
    return adata


def create_output_directory(result_params):
    os.makedirs(result_params['output_directory'], exist_ok=result_params['overwrite'])


def save_anndata(adobj, result_params):
    adobj.write( filename = op.join(result_params['output_directory'], result_params['adata_filename']))