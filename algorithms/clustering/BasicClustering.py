from algorithms.clustering.AbstractClusteringUpdate import ClusteringUpdateWrapper
from sklearn.cluster import KMeans

import pandas as pd

class BasicClustering(ClusteringUpdateWrapper):
    def __init__(self, data, n_clusters) -> None:
        super().__init__(data=data, n_clusters=n_clusters)
    


    def _compute_new_clustering(self) -> None:
        """
        Compute a new clustering based on the embedding computed in the previous step.
        """

        # store the old clustering
        self.data.obs['previous_clustering'] = self.data.obs['current_clustering'].copy()
        model = KMeans(n_clusters=self.data.uns['n_clusters'], random_state=0, max_iter=2)
        model.fit(self.data.obsm['embedding'])
        self.data.obs['current_clustering'] = model.labels_

    
    def _check_label_convergence(self, tolerance):
        return super()._check_label_convergence(tolerance)
    

    def _write_results(self, directory):
        
        print("Pass")