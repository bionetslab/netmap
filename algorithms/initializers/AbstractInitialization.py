from abc import ABC, abstractmethod
import os.path as op
import os
import time

class AbstractInitializer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _initialize_clustering(self) -> None:
        """
        This method defines how the intial clustering is obtained, e.g. read from file
        or random, ,.....

        Parameters:
        
        Returns: None (sets the clusterings)

        """

        pass


    
    def initialize_result_directory(self, output_directory):
        '''
        Initialize the result directory structure and automatically rename
        the directory if there is already a directory of the same name
        
        Arguments:
        output_directory: The fully qualified path of the output directory

        Returns:
        ---------------------------
        output_directory: The name of the created directory.
        '''

        if op.exists(output_directory):
            # if the directory exists create a time stamped directory
            # and return it
            timestamp = str(time.time())
            output_directory = f'{output_directory}_{timestamp}'

        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(op.join(output_directory, 'GRNs'))
        os.makedirs(op.join(output_directory, 'embedding'))
        os.makedirs(op.join(output_directory, 'clustering'))
        
        return output_directory