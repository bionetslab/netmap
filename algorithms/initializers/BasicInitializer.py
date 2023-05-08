from algorithms.initializers.AbstractInitialization import AbstractInitializer
from abc import abstractclassmethod

class BasicInitializer(AbstractInitializer):
    def __init__(self) -> None:
        super().__init__()


    @abstractclassmethod
    def _initialize_labels():
        """
        Method returns the intial cell labels, e.g. user defined from file

        Parameters:
        -------------------------

        Returns:
        -------------------------
        A numpy array matching the number of samples in the data
        """

        pass