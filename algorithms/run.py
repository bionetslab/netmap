from algorithms.BasicAlgorithm import BasicAlgorithm
from algorithms.Strategy import CellEmbeddingStrategy, ClusteringUpdateStategy, GRNInferrenceStrategy

ba = BasicAlgorithm(ClusteringUpdateStategy.BASIC, 
                    CellEmbeddingStrategy.BASIC,
                    GRNInferrenceStrategy.BASIC)