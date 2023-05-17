from algorithms.algorithms.AlgorithmWrapper import AlgorithmWrapper
from algorithms.Strategy import CellEmbeddingStrategy, ClusteringUpdateStategy, InitializationStrategy, GRNInferrenceStrategy
import pandas as pd
import numpy as np
import scanpy as sp
from algorithms.utils.data_utils import create_anndata_from_prefixes


class TestRunner:
    def __init__(self) -> None:
        pass


    
    def run_tests(self, data, n_clusters, max_iterations, output_directory):

        my_algorithm = AlgorithmWrapper(data,
                                        cell_embedding_strategy=CellEmbeddingStrategy.BASIC,
                                        clustering_strategy=ClusteringUpdateStategy.BASIC,
                                        grn_inference_strategy=GRNInferrenceStrategy.BASIC,
                                        initialization_strategy=InitializationStrategy.BASIC,
                                        max_iterations= max_iterations,
                                        output_directory = output_directory,
                                        n_clusters = n_clusters
                                        )
        my_algorithm.run(GRN_convergence_tolerance=1, cluster_convergence_tolerance=1)

        return my_algorithm
        

if __name__ == '__main__':

    filename = '/home/bionets-og86asub/Documents/netmap/data/tox-cd8'
    output = '/home/bionets-og86asub/Documents/netmap/temp-res/initial_results'
    prefix = ['GSM3568585_scRNA_D4.5_P14_Arm_1_']
    
    data = create_anndata_from_prefixes(data_directory = filename, prefix=prefix)
    test_runner = TestRunner()
    my_result = test_runner.run_tests(data, n_clusters=12, max_iterations=1, output_directory=output)

    
    
    
    from sklearn.metrics.cluster import contingency_matrix
    from networkx.algorithms.matching import max_weight_matching    
    import networkx as nx
    import os.path as op
    import scipy.sparse as scs
    cont = contingency_matrix(data.obs['previous_clustering'], data.obs['current_clustering'])
    G = nx.from_numpy_array(cont)
    max_weight_matching(G)

    filename = op.join(output, 'clustered_result.h5ad')
    data.write_h5ad(filename=filename)
    
    #self.GRN_inferrence._write_results(self.output_directory)
    #self.cell_embedding._write_results(self.output_directory)
    #self.clustering._write_results(self.output_directory)

    scs.save_npz(file= op.join(output, 'GRNs', 'cluster9.npz'), matrix = data.varp['iteration0_cluster9'])


    true_labels = [1,1,2,2,3,3, 6,6]
    predictions = [3,3,1,1, 2,2,6,6]

    import itertools

    # compute the weights for the matchin
    cont  = contingency_matrix(true_labels, predictions)
    a = [(index[0], index[1], item) for index, item in zip(itertools.product(range(cont.shape[0]), range(cont.shape[0], 2* cont.shape[0])), cont.flatten())]
    G = nx.Graph()
    G.add_weighted_edges_from(a)
    matching = list(max_weight_matching(G))
    d = cont.shape[0]
    label_matching = {}
    rownames = np.unique(true_labels)
    colnames = np.unique(predictions)
    for m in matching:
        if m[0]> m[1]:
            label_matching[rownames[m[0]-d]] = colnames[m[1]]
        else:
            label_matching[rownames[m[1]-d]] = colnames[m[0]]

    adjusted_cluster_labels = [label_matching[cl] for cl in ]