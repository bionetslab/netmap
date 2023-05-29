from algorithms.Strategy import GRNInferrenceStrategy
from algorithms.inference.BasicGRNInference import BasicGRNInference
from algorithms.inference.AracneGRNInference import AracneGRNInference
from algorithms.inference.AbstractGRNInferrence import AbstractGRNInferrence

class GRNInferenceFactory:
    def __init__(self) -> None:
        pass

    def create_inference_wrapper(self, type: GRNInferrenceStrategy, data) -> AbstractGRNInferrence:
        if type == GRNInferrenceStrategy.BASIC:
            return BasicGRNInference(data=data)
        if type == GRNInferrenceStrategy.ARACNE:
            return AracneGRNInference(data=data)
