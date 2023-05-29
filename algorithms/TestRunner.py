from algorithms.algorithms.AlgorithmWrapper import AlgorithmWrapper
import pandas as pd
import numpy as np
import scanpy as sp
from algorithms.utils.data_utils import create_anndata_from_prefixes
import os.path as op
import scipy.sparse as scs
import yaml
from algorithms.utils.config_parser import parse_configuration_file
import pandas as pd

class TestRunner:
    def __init__(self) -> None:
        pass
    
    def run_tests(self, data, configuration) -> AlgorithmWrapper:

        my_algorithm = AlgorithmWrapper(data, configuration)
        my_algorithm.run(GRN_convergence_tolerance=1, cluster_convergence_tolerance=1)

        return my_algorithm
        

if __name__ == '__main__':

    yml_config = '/home/bionets-og86asub/Documents/netmap/netmap-basic/config/aracne.yml'
    configuration = parse_configuration_file(yml_config)
    print(configuration)
    data = create_anndata_from_prefixes(data_directory = configuration['input.directory'], prefix=configuration['input.prefix'])
    test_runner = TestRunner()
    my_result = test_runner.run_tests(data=data, configuration=configuration)


    myexp = pd.DataFrame(my_result.data.X[my_result.data.obs['current_clustering'] == 11].toarray().T)
    myexp.index = my_result.data.var.index
    myexp.columns = my_result.data.obs[my_result.data.obs['current_clustering'] == 11].index

