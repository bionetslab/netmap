from abc import ABC, abstractmethod
import os.path as op
import pandas as pd
import numpy as np
import scipy as sc


class CellEmbeddingWrapper(ABC):
    def __init__(self, data) -> None:
        self.data = data

    @abstractmethod
    def _compute_new_cell_embedding(self) -> None:
        """
        Abstract method computing the new embedding of the data given the GRNs

        Parameters:
        ------------------------
        cluster_specific_GRNs: dictionary of cluster specific GRN modules, prefiltered

        """


    def run_embedding_step(self) -> None:
        self._compute_new_cell_embedding()
        

    @abstractmethod
    def _write_results(self) -> None:
        """
        Write all required results to file.

        """

        pass
