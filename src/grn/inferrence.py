import numpy as np
from captum.attr import GuidedBackprop, GradientShap, Deconvolution
import pingouin as pingu
import torch
from tqdm import tqdm

import pandas as pd

from netmap.src.utils.data_utils import attribution_to_anndata



def quantile_partitioning(data: np.ndarray, q: int) -> np.ndarray:
    """
    Performs quantile partitioning on a 1D NumPy array.

    The method orders the data and divides it into 'q' equal-sized partitions.
    A new array (mask) is created where each element is assigned a value
    of k/q, where k is the quantile that the element belongs to.

    Args:
        data (np.ndarray): A 1D NumPy array of numerical data.
        q (int): The number of quantiles to partition the data into. Must be a
                 positive integer.

    Returns:
        np.ndarray: A new array of the same shape as the input data, with
                    values representing the quantile partition.

    Raises:
        ValueError: If q is not a positive integer or if the input data is not
                    a 1D NumPy array.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input data must be a 1D NumPy array.")
    if not isinstance(q, int) or q <= 0:
        raise ValueError("The number of quantiles 'q' must be a positive integer.")

    n = len(data)
    if n == 0:
        return np.array([])
    
    # 1. Order data using argsort to get the indices that would sort the array
    sorted_indices = np.argsort(data)
    
    # 2. Cut into q equal pieces by calculating the size of each partition
    # We use float division to handle cases where n is not perfectly divisible by q.
    partition_size = n / q
    
    # 3. Initialize mask with same dimension as data
    mask = np.zeros_like(data, dtype=float)
    
    # 4. With k=current quantile add k/q to all cells belonging to k
    for k in range(1, q + 1):
        start_index = int((k - 1) * partition_size)
        end_index = int(k * partition_size)
        
        # Get the original indices that belong to the current quantile
        quantile_indices = sorted_indices[start_index:end_index]
        
        # Assign the value k/q to the corresponding positions in the mask
        mask[quantile_indices] = k / q
        
    return mask

def quantile_partitioning_2d(data: np.ndarray, q: int) -> np.ndarray:
    """
    Performs quantile partitioning row-wise on a 2D NumPy array using
    np.apply_along_axis for efficiency.

    Args:
        data (np.ndarray): A 2D NumPy array of numerical data.
        q (int): The number of quantiles to partition each row into.

    Returns:
        np.ndarray: A new 2D array with the same shape as the input,
                    where each row contains the quantile partitions.

    Raises:
        ValueError: If the input data is not a 2D NumPy array.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")

    # Use np.apply_along_axis to apply the 1D function to each row (axis=1).
    return np.apply_along_axis(quantile_partitioning, 1, data, q)

def compute_correlation_metric(data, cor_type):
    # Compute gene correlation measure
    #  'pingouin.pcorr', 'np.cov', 'np.corcoeff'
    if cor_type ==  'pingouin.pcorr':
        cov = pingu.pcorr(pd.DataFrame(data))
    elif cor_type == 'np.cov':
        cov = np.cov(data.T)
    elif cor_type == 'np.corrcoeff':
        cov = np.corrcoef(data.T)
    elif cor_type == 'None':
        cov = 1
    else: 
        cov = 1
    return cov

def aggregate_attributions(attributions, strategy = 'mean'):
    if strategy == 'mean':
        return np.mean(attributions, axis = 0)
    elif strategy == 'sum':
        return np.sum(attributions, axis = 0)
    elif strategy == 'median':
        return np.median(attributions, axis = 0)
    else:
        # Default to mean aggregation
        return np.mean(attributions, axis = 0)
    

def get_explainer(model, explainer_type, raw=False):
    if explainer_type in ['GuidedBackprop', 'Deconvolution']:
        explainer_mode = 'lrp-like'
    else:
        explainer_mode = 'shap-like'
    
        
    if explainer_type == 'GuidedBackprop': #fast
        explainer = GuidedBackprop(model)
    elif explainer_type == 'GradientShap': #fast
        if raw:
            explainer = GradientShap(model, multiply_by_inputs=False)
        else:
            explainer = GradientShap(model, multiply_by_inputs=True)

    elif explainer_type == 'Deconvolution': #fast
        explainer = Deconvolution(model)
    else:
        raise ValueError('no such method')
        
    return explainer, explainer_mode



def shuffle_each_column_independently(tensor):
    """
    Shuffles each column of a 2D PyTorch tensor independently.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: A new tensor with each of its columns independently shuffled.
    """
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional to shuffle columns.")

    # Create an empty tensor of the same size to store the shuffled columns
    shuffled_tensor = torch.empty_like(tensor)

    # Iterate through each column, shuffle it, and place it in the new tensor
    for i in range(tensor.size(1)):
        column = tensor[:, i]
        idx = torch.randperm(column.nelement())
        shuffled_tensor[:, i] = column[idx]

    return shuffled_tensor

def attribution_one_target( 
        target_gene,
        lrp_model,
        input_data,
        background,
        xai_type='lrp-like',
        randomize_background = False):
    
    attributions_list = []
    for m in range(len(lrp_model)):
        # Randomize backgorund for each round
        if randomize_background:
            background = shuffle_each_column_independently(background)

        model = lrp_model[m]
        #for _ in range(num_iterations):
        if xai_type == 'lrp-like':
            #print(input_data)
            #print(target_gene)
            attribution = model.attribute(input_data, target=target_gene)
                
        elif xai_type == 'shap-like':
            attribution = model.attribute(input_data, baselines = background, target = target_gene)

        attributions_list.append(attribution.detach().cpu().numpy())
    return attributions_list


def inferrence(models, data_train_full_tensor, gene_names, config):

    tms = []
    name_list = []
    target_names = []
    
    
    for trained_model in models:        
        trained_model.forward_mu_only = True
        explainer, xai_type = get_explainer(trained_model, config.xai_method, use_raw_attribution)
        tms.append(explainer)

    attributions = []

    ## ATTRIBUTIONS
    for g in tqdm(range(data_train_full_tensor.shape[1])):

        attributions_list = attribution_one_target(
            g,
            tms,
            data_train_full_tensor,
            data_train_full_tensor, #background data
            xai_type=xai_type,
            randomize_background = True)

        
        attributions_list = aggregate_attributions(attributions_list, strategy=config.aggregation_strategy)
        attributions.append(attributions_list)

    ## AGGREGATION: REPLACE LIST BY AGGREGATED DATA
    for i in range(len(attributions)):

        ## Create name vector
        name_list = name_list + list(gene_names)
        target_names = target_names+[gene_names[i]] *len(gene_names)



    attributions = np.hstack(attributions)
    
    index_list = [f"{s}_{t}" for (s, t) in zip(name_list, target_names)]
    cou = pd.DataFrame({'index': index_list, 'source':name_list, 'target':target_names})
    cou = cou.set_index('index')

    grn_adata = attribution_to_anndata(attributions, var=cou)

    return grn_adata
