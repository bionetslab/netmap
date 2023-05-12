from algorithms.algorithms.AlgorithmWrapper import AlgorithmWrapper
from algorithms.Strategy import CellEmbeddingStrategy, ClusteringUpdateStategy, InitializationStrategy, GRNInferrenceStrategy
import pandas as pd
import numpy as np
import scanpy as sp


class TestRunner:
    def __init__(self) -> None:
        pass


    
    def run_tests(self, data, n_clusters):

        my_algorithm = AlgorithmWrapper(data,
                                        cell_embedding_strategy=CellEmbeddingStrategy.BASIC,
                                        clustering_strategy=ClusteringUpdateStategy.BASIC,
                                        grn_inference_strategy=GRNInferrenceStrategy.BASIC,
                                        initialization_strategy=InitializationStrategy.BASIC,
                                        max_iterations= 12,
                                        n_clusters = n_clusters
                                        )
        my_algorithm.run(GRN_convergence_tolerance=1, cluster_convergence_tolerance=1)

        return my_algorithm
        

if __name__ == '__main__':

    filename = '/home/bionets-og86asub/Documents/netmap/data/tox-cd8'
    prefix = ['GSM3568585_scRNA_D4.5_P14_Arm_1_']
    from algorithms.utils.data_utils import create_anndata_from_prefixes
    data = create_anndata_from_prefixes(data_directory = filename, prefix=prefix)
    test_runner = TestRunner()
    test_runner.run_tests(data, n_clusters=12)

