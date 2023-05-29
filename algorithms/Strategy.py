from enum import Enum


class GRNInferrenceStrategy(Enum):
    BASIC = 1
    ARACNE = 2


class CellEmbeddingStrategy(Enum):
    BASIC = 1


class ClusteringUpdateStategy(Enum):
    BASIC = 1


class InitializationStrategy(Enum):
    BASIC = 1
