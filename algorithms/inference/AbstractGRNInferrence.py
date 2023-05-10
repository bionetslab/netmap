from abc import ABC, abstractmethod
import pandas as pd


class GRNInferrenceWrapper(ABC):
    def __init__(self,data) -> None:

        # The set of GRNs which have been infered in the previous iteration
        self.previous_grns = {}
        # The set of GRNs 
        self.current_grns = {}
        self.data = data 


        



    @abstractmethod        
    def _infer_cluster_specific_GRNS(self, cluster_labels) -> None:
        """
        This function returns a dictionary of GRNs, one for each label in cluster labels.
        The GRNs are not required to be a connected module. In general the GRN will be defined
        as the set of cluster specific edges (direction is important!) exluding the union of all
        edges which are part of another GRN.

        Parameters:
        ----------------
        expression_data: The expression data set as a pandas data frame (numeric) of dimension
            n * d where n is the number of samples and d is the number of features.
        cluster_labels: The cluster labels as a numpy array of length n. The row ordering must correspond
            to the orderin of the rows in the expression data set.
        
        The result will be stored in current_grns. 
        A pandas data frame containing the mandatory four columns + additional columns if required. 
        1. cluster_id
        2. source_node
        3. target_node
        4. edge_weight

        """
        pass
        
        
    @abstractmethod
    def _get_top_k_edges(self, k, cluster_labels, enforce_equal_k = False) -> pd.DataFrame:
        """
        Abstract method to returns the top k edges for the inferred GRNs 
        
        Parameters
        ----------

        k : Maximal number of edges to be returned.
        cluster_labels: The clusters ids in the current partition.
        enforce_equal_k: Require the number of edges k in each partition to be equal. 
            If set to True, the function must return k edges for each cluster. If this is not possible,
            k must be set to the maximal k of the smallest edge set. 
        Returns
        -------
         Returns:
        ------------------------
        A pandas data frame containing the top k edges for each cluster with the mandatory
        four columns + additional columns if required. 
        1. cluster_id
        2. source_node
        3. target_node
        4. edge_weight
        """

        pass


    @abstractmethod
    def _check_GRN_convergence(self, tolerance):
        """
        Abstract method checking whether a GRN has converged. It is recommended to implement this
        function in such a way, that convergence takes into account the edges, 
        in particular the directionality of the edges, if applicable. 

        Parameters:
        -----------------
        tolerance: a tolerance criterion specific for the convergence criterion. e.g. maximal
            number of edges allowed to be different between previous_grns and current_grns.

        Returns: 
        ----------------------
        True, if convergence has been reached within tolerance
        False, if convergence has not been reached.
        """

        pass


    def run_GRN_inference(self, cluster_labels, tolerance):
        self._infer_cluster_specific_GRNS(cluster_labels=cluster_labels)
        return self._check_GRN_convergence(tolerance=tolerance)





    @abstractmethod
    def _write_results(self):
        """
        Write all required results to file.

        """