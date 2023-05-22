from algorithms.Strategy import CellEmbeddingStrategy
from algorithms.embedding.BasicCellEmbedding import BasicCellEmbedding
import anndata as ad


class EmbeddingFactory:
    def create_embedding_wrapper(self, type: CellEmbeddingStrategy, data: ad.AnnData, **kwargs) -> CellEmbeddingStrategy:
        if type == CellEmbeddingStrategy.BASIC:
            return BasicCellEmbedding(data=data)
