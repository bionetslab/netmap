from algorithms.Strategy import CellEmbeddingStrategy
from algorithms.embedding.BasicCellEmbedding import BasicCellEmbedding
import anndata as ad
from algorithms.embedding.AbstractCellEmbedding import CellEmbeddingWrapper


class EmbeddingFactory:
    def create_embedding_wrapper(self, type: CellEmbeddingStrategy, data: ad.AnnData, **kwargs) -> CellEmbeddingWrapper:
        if type == CellEmbeddingStrategy.BASIC:
            return BasicCellEmbedding(data=data)
