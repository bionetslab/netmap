from abc import ABC, abstractmethod
from algorithms.Strategy import ClusteringUpdateStategy


class ClusteringUpdateWrapper(ABC):
    def __init__(self, data, n_clusters) -> None:
        self.previous_labels = {}
        self.current_labels = {}
        self.data = data
        self.n_clusters = n_clusters



    def get_current_labels(self) -> dict:
        return self.current_labels

    def ensure_consistent_labelling(self) -> None:
        """
        This method ensures that the cluster ids remain the same during the iterations.
        This is achieved using a ... matching.
        The labels are modified internally

        """

        #new_labels = 'magical new labels'
        #self.current_labels = new_labels
        pass
    
    @abstractmethod
    def _compute_new_clustering(self) -> None:
        """
        Compute a new partition (cell labels) based on the embedding computed in
        the previous algorithm step and store it in 'current_labels'

        Parameters:
        cluster_specific_GRNs: a dictionary of GRNS, one for each label

        """
        print('Compute new clustering')

        pass
        
    @abstractmethod
    def _check_label_convergence(self, tolerance):

        """
        Abstract method checking whether the labels have converged.

        Parameters:
        -----------------
        tolerance: a tolerance criterion specific for the convergence criterion. e.g. maximal
            number of cells allowed to be different between previous_grns and current_grns.

        Returns: 
        ----------------------
        True, if convergence has been reached within tolerance
        False, if convergence has not been reached.
        """

        print('Check label convergence')
        pass
    

    def run_clustering_step(self, tolerance):

        """
        Wrap all necessary steps for the clustering step in one convenience function.

        Parameters:
        ------------------------------
        cluster_specific_GRNs

        Returns:
        -------------------------------
        True, if converged; else False

        """
        self._compute_new_clustering()
        self.ensure_consistent_labelling()
        return self._check_label_convergence(tolerance=tolerance)
    


    @abstractmethod
    def _write_results(self):
        """
        Write all required results to file.
        
        """

        print('Writing results')