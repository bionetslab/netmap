from algorithms.embedding.AbstractCellEmbedding import CellEmbeddingWrapper
import anndata as ad


class BasicCellEmbedding(CellEmbeddingWrapper):
    def __init__(self, data:ad.AnnData) -> None:
        super().__init__(data=data)

    def _write_results(self):
        return super()._write_results()
    

    def _compute_new_cell_embedding(self, cluster_specific_GRNs):
        return super()._compute_new_cell_embedding(cluster_specific_GRNs)
    
    