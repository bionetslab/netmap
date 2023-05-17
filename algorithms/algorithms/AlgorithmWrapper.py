from abc import ABC, abstractmethod
from algorithms.Strategy import CellEmbeddingStrategy, ClusteringUpdateStategy, GRNInferrenceStrategy
from algorithms.clustering.ClusteringFactory import ClusteringFactory
from algorithms.embedding.EmbeddingFactory import EmbeddingFactory
from algorithms.inference.GRNInferenceFactory import GRNInferenceFactory
from algorithms.initializers.InitializerFactory import InitializerFactory
from algorithms.Strategy import InitializationStrategy
import pandas as pd
import logging
import os.path as op
import anndata as ad

class AlgorithmWrapper(ABC):
    def __init__(self, 
                 data:ad.AnnData,
                cell_embedding_strategy: CellEmbeddingStrategy, 
                clustering_strategy: ClusteringUpdateStategy, 
               grn_inference_strategy: GRNInferrenceStrategy,
               initialization_strategy: InitializationStrategy,
               max_iterations = 100,
               n_clusters=12,
               output_directory='/tmp/netmap') -> None:
        
        self.logger = logging.getLogger('algolog')
        self.logger.setLevel(logging.INFO)
        self.data = data
        self.max_iterations = max_iterations
        self.n_cluster=n_clusters
        self.output_directory = output_directory

        # save the variables
        self.cell_embedding_strategy = cell_embedding_strategy
        self.clustering_strategy = clustering_strategy
        self.grn_inference_strategy = grn_inference_strategy
        self.initialization_strategy = initialization_strategy

        # initialize 

        self.initializer = InitializerFactory().create_initializer_wrapper(type=initialization_strategy, 
                                                                           data = data, 
                                                                           n_clusters=n_clusters, 
                                                                           max_iterations=max_iterations)
        
        self.GRN_inferrence = GRNInferenceFactory().create_inference_wrapper(type=grn_inference_strategy, data =data)
        self.cell_embedding = EmbeddingFactory().create_embedding_wrapper(type=cell_embedding_strategy, data=data)
        self.clustering = ClusteringFactory().create_clustering_wrapper(type=clustering_strategy, data=data)
        


    def check_convergence(self, grn_convergence, label_convergence):
        """
        Require the labels and the GRNs to be converged.
        """
        return (grn_convergence and label_convergence)
    

    def run(self, GRN_convergence_tolerance, cluster_convergence_tolerance) -> None:

        """
        Run method for an algorithm. This should wrap everything from initialization to 
        results production.


        """
        print('Running algorithm')
        print('Initializing Algorithm')
        self.output_directory = self.initializer.initialize_result_directory(self.output_directory)
        self.initializer._initialize_clustering()
        print('Finshed setting up')

        
        print('Start inference')
        label_convergence = False
        grn_convergence = False
        iterations = 0
        while not self.check_convergence(grn_convergence, label_convergence) and self.data.uns['current_iteration']<self.data.uns['max_iterations']:
            print('GRN inferrence')
            grn_convergence = self.GRN_inferrence.run_GRN_inference(consistency=GRN_convergence_tolerance)
            print('Embedding')
            self.cell_embedding.run_embedding_step()
            print('Clustering')
            label_convergence = self.clustering.run_clustering_step(tolerance=cluster_convergence_tolerance)
            iterations = iterations+1

            self.data.uns['current_iteration'] = self.data.uns['current_iteration'] +1

        print('Finished inference')
        
        print('Writing results to file')
        self.write_results()
        print('Finished')

            
        

        
    def write_results(self):
        """
        Implement some code to write the results.
        By default the h5ad formatted Anndata is stored. Additional functionalities can 
        be implemented for each component.
        """

        filename = op.join(self.output_directory, 'clustered_result.h5ad')
        self.data.write_h5ad(filename=filename)
        
        self.GRN_inferrence._write_results()
        self.cell_embedding._write_results()
        self.clustering._write_results()