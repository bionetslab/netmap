from abc import ABC, abstractmethod
import os.path as op
import os
import time


class AbstractInitializer(ABC):
    def __init__(self, data) -> None:
        self.data = data

    @abstractmethod
    def _initialize_clustering(self) -> None:
        """
        This method defines how the intial clustering is obtained, e.g. read from file
        or random, ,.....

        Parameters:

        Returns: None (sets the clusterings)

        """

        pass

    def initialize_result_directory(self) -> None:
        """
        Initialize the result directory structure and automatically rename
        the directory if there is already a directory of the same name

        Arguments:
        output_directory: The fully qualified path of the output directory

        Returns:
        ---------------------------
        output_directory: The name of the created directory.
        """

        print('Initializing result directory')
        if op.exists(self.data.uns['output.directory']):
            # if the directory exists create a time stamped directory
            # and return it
            timestamp = str(time.time())
            output_directory = f"{self.data.uns['output.directory']}_{timestamp}"
            self.data.uns['output.directory'] = output_directory

        os.makedirs(self.data.uns['output.directory'], exist_ok=True)
        os.makedirs(op.join(self.data.uns['output.directory'], "GRNs"))
        os.makedirs(op.join(self.data.uns['output.directory'], "embedding"))
        os.makedirs(op.join(self.data.uns['output.directory'], "clustering"))

        self.data.uns["GRN_dir"] = op.join(self.data.uns['output.directory'], "GRNs")
        self.data.uns["embedding_dir"] = op.join(self.data.uns['output.directory'], "embedding")
        self.data.uns["clustering_dir"] = op.join(self.data.uns['output.directory'], "clustering")
