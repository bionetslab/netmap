from abc import ABC, abstractmethod


class InitializationWrapper(ABC):
    def __init__(self) -> None:
        self.name = "InitializationWrapper"
        self.clustering = None
    
    def initialize_clustering(self, clustering):
        self.clustering = clustering


    