from abc import ABC, abstractmethod
import os.path as op
import pandas as pd
import numpy as np
import scipy as sc


class CellEmbeddingWrapper(ABC):
    def __init__(self, data) -> None:
        self.data = data
        self.embedding = None


    @abstractmethod
    def _compute_new_cell_embedding(self, cluster_specific_GRNs):
        """
        Abstract method computing the new embedding of the data given the GRNs

        Parameters:
        ------------------------
        cluster_specific_GRNs: dictionary of cluster specific GRN modules, prefiltered

        """


    def get_embedding(self):
        return self.embedding
    

    def run_embedding_step(self, cluster_specific_GRNs):
        self._compute_new_cell_embedding(cluster_specific_GRNs)
        return self.embedding
    


    @abstractmethod
    def _write_results(self):
        """
        Write all required results to file.
        
        """

        pass

