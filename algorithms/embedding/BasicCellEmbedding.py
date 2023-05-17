from algorithms.embedding.AbstractCellEmbedding import CellEmbeddingWrapper
import anndata as ad
import numpy as np
import scipy.sparse as scs
import os.path as op

class BasicCellEmbedding(CellEmbeddingWrapper):
    def __init__(self, data:ad.AnnData) -> None:
        super().__init__(data=data)

    def _write_results(self, directory):
        """
        Dummy write method. Probably useless as is.
        """
        # try:
        #     file_path = op.join(directory, 'embedding.tsv')
        #     self.data.obsm['embedding'].X.to_csv(file_path, sep = '\t')
        # except KeyError:
        #     print('Results not initialized.')
        pass


    def _compute_new_cell_embedding(self):
        """
        Computes a new cell embedding. 
        The basic cell embedding is simply the union of all
        genes contained in all the GRNs. The embedding is stored in the 
        obsm field of the anndata object.

        """

        i = self.data.uns['current_iteration']
        all_genes = []
        # select all genes in all GRNs as the current embedding.
        for GRN in self.data.uns['GRNs'][f'iteration_{i!r}'].values():
            all_genes.append(np.unique(self.data.varp[GRN].nonzero()))

        all_genes = np.unique(all_genes)
        # there should just be one embedding, so we don't need to to some
        # kind of storage yoga
        self.data.obsm['embedding'] = self.data[:, all_genes].X.copy()
            
        
        