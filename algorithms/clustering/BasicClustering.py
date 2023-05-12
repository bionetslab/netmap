from algorithms.clustering.AbstractClusteringUpdate import ClusteringUpdateWrapper
from sklearn.cluster import SpectralBiclustering



class BasicClustering(ClusteringUpdateWrapper):
    def __init__(self, data, n_clusters) -> None:
        super().__init__(data=data, n_clusters=n_clusters)
    


    def _compute_new_clustering(self, embedding) -> None:

        model = SpectralBiclustering(n_clusters=self.n_clusters, method="log", random_state=0)
        model.fit(embedding)

        return
    
    def _check_label_convergence(self, tolerance):
        return super()._check_label_convergence(tolerance)
    

    def _write_results(self):
        return super()._write_results()