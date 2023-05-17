from algorithms.initializers.AbstractInitialization import AbstractInitializer
from abc import abstractclassmethod
import anndata as ad
import scanpy as sp
from algorithms.utils.expceptions import NotInitializerError
import numpy as np

class BasicInitializer(AbstractInitializer):
    def __init__(self, data:ad.AnnData, n_clusters:int, max_iterations:int) -> None:
        super().__init__()
        self.data = data # This is initialized in a second step.
        
        # These are some house keeping variables that are required for the 
        # entire algorithms. 
        self.data.uns['n_clusters'] = n_clusters
        self.data.uns['max_iterations'] = max_iterations
        self.data.uns['current_iteration']  = 0
        data.uns['GRNs'] = {}


    def _initialize_clustering(self):
        """
        Method creates the the intial cell labels, in this case the labels
        are generated uniformly at random from the number of clusters.

        Parameters:
        -------------------------

        Returns:
        -------------------------
        A numpy array matching the number of samples in the data
        """
        if self.data is None:
            raise NotInitializerError("Data object not initialized")
        
        # sample uniformly at random from the number of clusters
        random_clusters = np.random.choice(self.data.uns['n_clusters'], self.data.X.shape[0])
        # add a column with the initial clustering
        self.data.obs['initial_clustering'] = random_clusters
        self.data.obs['current_clustering'] = random_clusters

    
        

   
