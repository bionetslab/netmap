from abc import ABC, abstractmethod
import pandas as pd


class AbstractGRNInferrence(ABC):
    def __init__(self,data) -> None:

        # The set of GRNs which have been infered in the previous iteration
        self.data = data 


        



    @abstractmethod        
    def _infer_cluster_specific_GRNS(self, cluster_labels) -> None:
        """
        This function infers the GRNs, one for each label in cluster labels.
        The GRNs are not required to be a connected module. In general the GRN will be defined
        as the set of cluster specific edges exluding the union of all edges which are part of another GRN.

        This method infers the new GRNs and stores them in the current GRN slot f the data.
        The previous' iterations GRNs need to be pushed into the respective slot.

        Parameters:

        Returns:
        ----------------
        The result will be stored the data:AnnData object

        AnnData object has allows the storage of Graph like data in obsp and varp, as well
        as the storage of unstructured data in uns.

        The following storage is expected: (This is similar to how scanpy stores it's umaps)

        The keys for the GRNs should be stored in the .uns dictonary
        the key follows the follwing convention.

        
        self.data.uns['GRNs'] = {cluster_id: {"GRN_key": <GRN_key>}}
        self.data.uns['old_GRN'] = self.data.uns['new_GRNs'].copy()

        # For each cluster a different sparse GRN 
        self.data.obsp[<GRN_key>] = sc.sparse_crs(GRN)
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
        
        """

        print('Get top k edges')


    @abstractmethod
    def _check_GRN_convergence(self, consistency):
        """
        Abstract method checking whether a GRN has converged. It is recommended to implement this
        function in such a way, that convergence takes into account the edges, 
        in particular the directionality of the edges, if applicable. 

        Parameters:
        -----------------
        consistency: a tolerance criterion specific for the convergence criterion. e.g. maximal
            number of edges allowed to be different between previous_grns and current_grns.

        Returns: 
        ----------------------
        True, if convergence has been reached within tolerance
        False, if convergence has not been reached.
        """

        print('Check GRN convergence')


    def run_GRN_inference(self, consistency):
        self._infer_cluster_specific_GRNS()
        return self._check_GRN_convergence(consistency=consistency)





    @abstractmethod
    def _write_results(self):
        """
        Write all required results to file.

        """

        print('Results')