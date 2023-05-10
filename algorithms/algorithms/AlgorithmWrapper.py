from abc import ABC, abstractmethod
from algorithms.Strategy import CellEmbeddingStrategy, ClusteringUpdateStategy, GRNInferrenceStrategy
from algorithms.inference.AbstractGRNInferrence import GRNInferrenceWrapper
from algorithms.embedding.AbstractCellEmbedding import CellEmbeddingWrapper
from algorithms.clustering.AbstractClusteringUpdate import ClusteringUpdateWrapper
from algorithms.clustering.ClusteringFactory import ClusteringFactory
from algorithms.embedding.EmbeddingFactory import EmbeddingFactory
from algorithms.inference.GRNInferenceFactory import GRNInferenceFactory
from algorithms.initializers.InitializerFactory import InitializerFactory
from algorithms.Strategy import InitializationStrategy

class AlgorithmWrapper(ABC):
    def __init__(self, 
                 data,
               cell_embedding_strategy: CellEmbeddingStrategy, 
                clustering_strategy: ClusteringUpdateStategy, 
               grn_inference_strategy: GRNInferrenceStrategy,
               initialization_strategy: InitializationStrategy,
               max_iterations = 100) -> None:
        

        self.data = data
        self.max_iterations = max_iterations

        # save the variables
        self.cell_embedding_strategy = cell_embedding_strategy
        self.clustering_strategy = clustering_strategy
        self.grn_inference_strategy = grn_inference_strategy

        # initialize 
        self.GRN_inferrence = GRNInferenceFactory().create_inference_wrapper(type=grn_inference_strategy)
        self.cell_embedding = EmbeddingFactory().create_embedding_wrapper(type=cell_embedding_strategy)
        self.clustering = ClusteringFactory().create_clustering_wrapper(type=clustering_strategy)

        self.initializer = InitializerFactory().create_initializer_wrapper(type=initialization_strategy)
    
    def check_convergence(self, grn_convergence, label_convergence):
        """
        Require the labels and the GRNs to be converged.
        """
        return (grn_convergence and label_convergence)
    

    def run(self, GRN_convergence_tolerance, cluster_convergence_tolerance):
        self.initializer.initialize_clustering()
        initial_clustering  = self.initializer.get_initial_clustering()
        

        label_convergence = False
        grn_convergence = False
        iterations = 0
        while self.check_convergence(grn_convergence, label_convergence) and iterations<self.max_iterations:
            grn_convergence = self.GRN_inferrence.run_GRN_inference(cluster_labels=initial_clustering,
                                                  tolerance=GRN_convergence_tolerance)
            self.cell_embedding.run_embedding_step(cluster_specific_GRNs=self.GRN_inferrence.current_grns)
            label_convergence = self.clustering.run_clustering_step(tolerance=cluster_convergence_tolerance, 
                                                embedding=self.cell_embedding)
            
        self.write_results()
        
            
        

        
    def write_results(self):
        """
        Implement some code to write the results
        """

        self.GRN_inferrence._write_results()
        self.cell_embedding._write_results()
        self.clustering._write_results()