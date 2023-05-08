from algorithms.Strategy import CellEmbeddingStrategy
from algorithms.embedding.BasicCellEmbedding import BasicCellEmbedding

class EmbeddingFactory:

    def create_embedding_wrapper(self, type:CellEmbeddingStrategy):
        if type == CellEmbeddingStrategy.BASIC:
            return BasicCellEmbedding()
        

        