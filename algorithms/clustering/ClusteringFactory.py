from algorithms.Strategy import ClusteringUpdateStategy
from algorithms.clustering.BasicClustering import BasicClustering
from algorithms.clustering.AbstractClusteringUpdate import ClusteringUpdateWrapper

class ClusteringFactory:
    
    def create_clustering_wrapper(self, type: ClusteringUpdateStategy, **kwargs) -> ClusteringUpdateWrapper:
        """
        Instantiates the correct Clustering Strategy based on the paramters.
        
        """
        if type == ClusteringUpdateStategy.BASIC:
            return BasicClustering(**kwargs)
        

