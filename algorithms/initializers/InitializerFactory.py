from algorithms.initializers.BasicInitializer import BasicInitializer
from algorithms.Strategy import InitializationStrategy

class InitializerFactory:
    def __init__(self) -> None:
        pass


    def create_initializer_wrapper(self, type:InitializationStrategy, **kwargs):
        if type == InitializationStrategy.BASIC:
            return BasicInitializer(**kwargs)