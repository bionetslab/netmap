from algorithms.Strategy import GRNInferrenceStrategy
from algorithms.inference.BasicGRNInference import BasicGRNInference


class GRNInferenceFactory:
    def __init__(self) -> None:
        pass

    def create_inference_wrapper(self, type: GRNInferrenceStrategy, data):
        if type == GRNInferrenceStrategy.BASIC:
            return BasicGRNInference(data=data)
