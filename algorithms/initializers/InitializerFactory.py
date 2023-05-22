from algorithms.initializers.BasicInitializer import BasicInitializer
from algorithms.Strategy import InitializationStrategy
from anndata import AnnData
from algorithms.initializers.AbstractInitialization import AbstractInitializer

class InitializerFactory:
    def __init__(self) -> None:
        pass

    def create_initializer_wrapper(self, type: InitializationStrategy, data: AnnData, **kwargs) -> AbstractInitializer:
        if type == InitializationStrategy.BASIC:
            return BasicInitializer(data=data, **kwargs)
