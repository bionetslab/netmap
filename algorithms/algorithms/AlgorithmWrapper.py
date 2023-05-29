from abc import ABC, abstractmethod
from algorithms.Strategy import (
    CellEmbeddingStrategy,
    ClusteringUpdateStategy,
    GRNInferrenceStrategy,
)
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
    def __init__(
        self,
        data: ad.AnnData,
        configuration:dict
    ) -> None:

        self.data = data

        self.initializer = InitializerFactory().create_initializer_wrapper(
            type=InitializationStrategy[configuration['strategy.InitializationStrategy']],
            data=data,
            configuration=configuration)

        self.GRN_inferrence = GRNInferenceFactory().create_inference_wrapper(type=GRNInferrenceStrategy[configuration['strategy.GRNInferrenceStrategy']], data=data)
        self.cell_embedding = EmbeddingFactory().create_embedding_wrapper(type=CellEmbeddingStrategy[configuration['strategy.CellEmbeddingStrategy']], data=data)
        self.clustering = ClusteringFactory().create_clustering_wrapper(type=ClusteringUpdateStategy[configuration['strategy.ClusteringUpdateStategy']], data=data)

    def check_convergence(self, grn_convergence, label_convergence) -> bool:
        """
        Require the labels and the GRNs to be converged.
        """
        return grn_convergence and label_convergence

    def run(self, GRN_convergence_tolerance, cluster_convergence_tolerance) -> None:

        """
        Run method for an algorithm. This should wrap everything from initialization to
        results production.


        """
        print("Running algorithm")
        print("Initializing Algorithm")
        self.initializer._initialize_clustering()
        self.initializer.initialize_result_directory()

        print(self.data.uns)
        print("Finshed setting up")

        print("Start inference")
        label_convergence = False
        grn_convergence = False
        iterations = 0
        while (
            not self.check_convergence(grn_convergence, label_convergence)
            and self.data.uns["current_iteration"] < self.data.uns["algorithm.max_iterations"]
        ):
            print("GRN inferrence")
            grn_convergence = self.GRN_inferrence.run_GRN_inference(consistency=GRN_convergence_tolerance)
            print("Embedding")
            self.cell_embedding.run_embedding_step()
            print("Clustering")
            label_convergence = self.clustering.run_clustering_step(tolerance=cluster_convergence_tolerance)
            iterations = iterations + 1
            self.data.uns["current_iteration"] = self.data.uns["current_iteration"] + 1

        print("Finished inference")

        print("Writing results to file")
        self.write_results()
        print("Finished")

    def write_results(self) -> None:
        """
        Implement some code to write the results.
        By default the h5ad formatted Anndata is stored. Additional functionalities can
        be implemented for each component.
        """

        filename = op.join(self.data.uns['output.directory'], "clustered_result.h5ad")
        self.data.write_h5ad(filename=filename)

        self.GRN_inferrence._write_results()
        self.cell_embedding._write_results()
        self.clustering._write_results()
