from algorithms.initializers.BasicInitializer import BasicInitializer
from algorithms.Strategy import InitializationStrategy
from anndata import AnnData
class InitializerFactory:
    def __init__(self) -> None:
        pass


    def create_initializer_wrapper(self, type:InitializationStrategy, data:AnnData, **kwargs):
        if type == InitializationStrategy.BASIC:
            return BasicInitializer(data=data, **kwargs)