from abc import ABC, abstractmethod
from algorithms.Strategy import ClusteringUpdateStategy
from sklearn.metrics.cluster import contingency_matrix
from networkx.algorithms.matching import max_weight_matching
import networkx as nx
import itertools
import numpy as np
from algorithms.utils.expceptions import InconsistenClusterExpection

class ClusteringUpdateWrapper(ABC):
    def __init__(self, data) -> None:
        self.data = data


    def ensure_consistent_labelling(self) -> None:
        """
        This method ensures that the cluster ids remain the same during the iterations.
        This is achieved using a maximum weight matching with the Hungarian algorithm
        The labels are updated so that the clusters are aligned with the initial clustering.

        1. Compute a contingency matrix.
        2. find a maximum weight matching between the old labels and the new lables
        3. Create a dictionary where the new label maps to the old label
        4. Update the new labels to be consitent with the old labels

        """
        cont = contingency_matrix(self.data.obs['previous_clustering'], 
                                                     self.data.obs['current_clustering'])
        
        a = [(index[0], index[1], item) for index, item in zip(itertools.product(range(cont.shape[0]), range(cont.shape[0], 2* cont.shape[0])), cont.flatten())]
        G = nx.Graph()
        G.add_weighted_edges_from(a)
        matching = list(max_weight_matching(G))
        d = cont.shape[0]
        label_matching = {}
        
        rownames = np.unique(self.data.obs['previous_clustering'])
        colnames = np.unique(self.data.obs['current_clustering'])
        for m in matching:
            if m[0]> m[1]:
                label_matching[rownames[m[0]-d]] = colnames[m[1]]
            else:
                label_matching[rownames[m[1]-d]] = colnames[m[0]]
        
        
        # Fallback if there is not an optimal cluster correspondence
        for c in colnames:
            if c not in label_matching.keys():
                raise InconsistenClusterExpection("Number of clusters changes")

        adjusted_cluster_labels = [label_matching[cl] for cl in self.data.obs['current_clustering']]
        self.data.obs['current_clustering'] = adjusted_cluster_labels


        
    
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