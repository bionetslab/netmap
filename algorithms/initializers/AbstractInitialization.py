from abc import ABC, abstractmethod


class AbstractInitializer(ABC):
    def __init__(self) -> None:
        self.name = "InitializationWrapper"
        self.clustering = None
    

    @abstractmethod
    def initialize_clustering(self) -> None:
        """
        This method defines how the intial clustering is obtained, e.g. read from file
        or random, ,.....

        Parameters:
        
        Returns: None (sets the clusterings)

        """



    def get_initial_clustering(self):
        return self.clustering